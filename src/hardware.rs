//! Hardware probing and empirically-grounded automatic retriever selection.
//!
//! Real-world desktop workloads (e.g. ~30 Chrome tabs + IDE + chat/editor + OS)
//! routinely occupy 7-11 GiB of RAM before MinSync runs. Measured on macOS VMs
//! (8 / 16 / 24 GiB with 30 Chrome tabs + desktop apps) and x86_64 CPU-only
//! Linux (no Metal/CUDA/Accelerate):
//!
//! | Retriever | Peak RSS (Metal / CPU-only) | Safe host spec under desktop load |
//! |---|---|---|
//! | BM25 (`lindera` + `bm25-index.inv`) | ~276 MB sync / ~165 MB query | Every machine (incl. 4-8 GiB) |
//! | SPLADE (`KURE-Sparse-nano` 0.1B) | ~626 MB Metal / ~1.0 GB CPU | RAM >= 16 GiB (and GPU or >= 4 cores) |
//! | Dense (`Qwen3-Embedding-0.6B`) | ~2.3 GB Metal / ~3.8 GB CPU | Apple Silicon >= 24 GiB RAM, or NVIDIA CUDA >= 8 GB VRAM + >= 16 GiB RAM |
//!
//! `minsync init` uses [`Hardware::probe`] whenever `--embedder auto` or
//! `--sparse auto` is chosen (the default), so low-spec PCs never crash or
//! swap-thrash trying to load neural models, while capable machines enable the
//! full multi-retriever stack. Explicit flags always override the probe.

use crate::config::{DEFAULT_EMBEDDER_ID, DEFAULT_SPARSE_ID, NONE_EMBEDDER_ID, NONE_SPARSE_ID};
use serde::{Deserialize, Serialize};
use std::process::Command;

pub const GIB: u64 = 1024 * 1024 * 1024;

/// Minimum unified memory on Apple Silicon for auto-enabling local dense embeddings.
pub const DENSE_MIN_APPLE_RAM_GIB: u64 = 24;
/// Minimum host RAM alongside a CUDA GPU for auto-enabling local dense embeddings.
pub const DENSE_MIN_CUDA_HOST_RAM_GIB: u64 = 16;
/// Minimum NVIDIA VRAM (in MiB; 8 GB cards report ~8188-8192 MiB, sold as 8000 MB).
pub const DENSE_MIN_CUDA_VRAM_MIB: u64 = 8_000;
/// Minimum host RAM for auto-enabling SPLADE (0.1B) under realistic desktop load.
pub const SPLADE_MIN_RAM_GIB: u64 = 16;
/// Minimum logical CPU cores for auto-enabling CPU-only SPLADE.
pub const SPLADE_MIN_CPU_CORES: usize = 4;

/// Environment variable overriding hardware detection with a JSON [`Hardware`]
/// payload (used in deterministic tests and manual QA).
pub const HARDWARE_OVERRIDE_ENV: &str = "MINSYNC_HARDWARE_OVERRIDE";

/// NVIDIA CUDA GPU discovered on Windows / Linux.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CudaGpu {
    pub name: String,
    pub vram_mib: u64,
}

/// Host hardware characteristics relevant to indexing and query-time footprint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Hardware {
    /// OS family (`"macos"`, `"windows"`, `"linux"`, ...).
    pub os: String,
    /// CPU architecture (`"aarch64"`, `"x86_64"`, ...).
    pub arch: String,
    /// Logical CPU cores (`std::thread::available_parallelism`).
    pub cores: usize,
    /// Total physical RAM in bytes (`None` if unreadable).
    pub total_ram_bytes: Option<u64>,
    /// CUDA-capable NVIDIA GPUs (Windows / Linux); empty when absent.
    #[serde(default)]
    pub cuda_gpus: Vec<CudaGpu>,
}

/// Recommended local retriever tier derived from [`Hardware`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HardwareProfile {
    /// Dense (`Qwen3-Embedding-0.6B`) + SPLADE (`KURE-Sparse-nano`) + BM25.
    Full,
    /// SPLADE (`KURE-Sparse-nano`) + BM25 (dense skipped to protect RAM/CPU).
    SparseAndBm25,
    /// BM25 only (zero ML inference — safe on 4-8 GiB or low-core CPU-only PCs).
    Bm25Only,
}

impl HardwareProfile {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::SparseAndBm25 => "sparse+bm25",
            Self::Bm25Only => "bm25-only",
        }
    }

    /// Default `(embedder_id, sparse_id)` pair for this profile. BM25 is always on.
    pub fn retriever_ids(self) -> (&'static str, &'static str) {
        match self {
            Self::Full => (DEFAULT_EMBEDDER_ID, DEFAULT_SPARSE_ID),
            Self::SparseAndBm25 => (NONE_EMBEDDER_ID, DEFAULT_SPARSE_ID),
            Self::Bm25Only => (NONE_EMBEDDER_ID, NONE_SPARSE_ID),
        }
    }
}

impl Hardware {
    /// Probe the current machine (honoring [`HARDWARE_OVERRIDE_ENV`] if set).
    pub fn probe() -> Self {
        if let Ok(raw) = std::env::var(HARDWARE_OVERRIDE_ENV) {
            let trimmed = raw.trim();
            if !trimmed.is_empty() {
                if let Ok(parsed) = serde_json::from_str::<Hardware>(trimmed) {
                    return parsed;
                }
            }
        }
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let total_ram_bytes = total_ram_bytes();
        let cuda_gpus = if matches!(os.as_str(), "windows" | "linux") {
            probe_cuda_gpus(&os)
        } else {
            Vec::new()
        };
        Self {
            os,
            arch,
            cores,
            total_ram_bytes,
            cuda_gpus,
        }
    }

    /// True on Apple Silicon Macs (Metal GPU + Apple Neural Engine).
    pub fn is_apple_silicon(&self) -> bool {
        self.os == "macos" && self.arch == "aarch64"
    }

    /// Largest NVIDIA CUDA VRAM in MiB, if any.
    pub fn max_cuda_vram_mib(&self) -> u64 {
        self.cuda_gpus.iter().map(|g| g.vram_mib).max().unwrap_or(0)
    }

    /// True when the machine has a hardware accelerator (Apple Silicon or CUDA GPU).
    pub fn has_accelerator(&self) -> bool {
        self.is_apple_silicon() || !self.cuda_gpus.is_empty()
    }

    /// Recommend the safest high-quality local retriever tier for this hardware.
    pub fn recommend(&self) -> HardwareProfile {
        let Some(ram_bytes) = self.total_ram_bytes else {
            // Unreadable RAM: avoid loading the heavy 0.6B dense model by accident.
            return if self.has_accelerator() || self.cores >= SPLADE_MIN_CPU_CORES {
                HardwareProfile::SparseAndBm25
            } else {
                HardwareProfile::Bm25Only
            };
        };

        // 1. Can this machine comfortably run local dense (0.6B) + SPLADE + BM25?
        if self.is_apple_silicon() && ram_bytes >= DENSE_MIN_APPLE_RAM_GIB * GIB {
            return HardwareProfile::Full;
        }
        if self.max_cuda_vram_mib() >= DENSE_MIN_CUDA_VRAM_MIB
            && ram_bytes >= DENSE_MIN_CUDA_HOST_RAM_GIB * GIB
        {
            return HardwareProfile::Full;
        }

        // 2. Can it comfortably run SPLADE (0.1B) + BM25 without starving desktop apps?
        if ram_bytes >= SPLADE_MIN_RAM_GIB * GIB
            && (self.has_accelerator() || self.cores >= SPLADE_MIN_CPU_CORES)
        {
            return HardwareProfile::SparseAndBm25;
        }

        // 3. Low RAM (< 16 GiB) or low-core CPU-only: zero-ML BM25 only.
        HardwareProfile::Bm25Only
    }

    /// Human-readable one-line hardware summary.
    pub fn summary(&self) -> String {
        let ram = match self.total_ram_bytes {
            Some(bytes) => format!("{:.1} GiB RAM", bytes as f64 / GIB as f64),
            None => "unknown RAM".to_string(),
        };
        let accel = if self.is_apple_silicon() {
            "Apple Silicon (Metal/ANE)".to_string()
        } else if let Some(gpu) = self.cuda_gpus.iter().max_by_key(|g| g.vram_mib) {
            format!(
                "CUDA {} ({:.1} GiB VRAM)",
                gpu.name,
                gpu.vram_mib as f64 / 1024.0
            )
        } else {
            "CPU-only".to_string()
        };
        format!(
            "{}/{}, {}, {} cores, {}",
            self.os, self.arch, ram, self.cores, accel
        )
    }

    /// Plain explanation of why [`Self::recommend`] picked its profile.
    pub fn rationale(&self) -> String {
        match self.recommend() {
            HardwareProfile::Full => {
                "dense + SPLADE + BM25 (machine has enough RAM and GPU/Apple Silicon headroom)"
                    .to_string()
            }
            HardwareProfile::SparseAndBm25 => {
                if self.is_apple_silicon() {
                    format!(
                        "SPLADE + BM25 (dense 0.6B skipped below {DENSE_MIN_APPLE_RAM_GIB} GiB unified memory to avoid desktop swap)"
                    )
                } else {
                    format!(
                        "SPLADE + BM25 (dense 0.6B skipped without an {}-GB+ CUDA GPU)",
                        DENSE_MIN_CUDA_VRAM_MIB / 1000
                    )
                }
            }
            HardwareProfile::Bm25Only => {
                if self
                    .total_ram_bytes
                    .is_some_and(|b| b < SPLADE_MIN_RAM_GIB * GIB)
                {
                    format!(
                        "BM25 only (RAM < {SPLADE_MIN_RAM_GIB} GiB; skipping neural models to avoid desktop memory pressure/swap)"
                    )
                } else {
                    format!(
                        "BM25 only (CPU-only with < {SPLADE_MIN_CPU_CORES} cores; skipping neural inference)"
                    )
                }
            }
        }
    }
}

/// Resolve an `init` `--embedder` or `--sparse` flag: `"auto"` uses the
/// probed hardware profile; any explicit value (`none`, `native:...`, `tei:...`,
/// `openai:...`) is kept verbatim.
pub fn resolve_auto_retrievers(
    embedder: &str,
    sparse: &str,
    hardware: &Hardware,
) -> (String, String) {
    let (auto_embedder, auto_sparse) = hardware.recommend().retriever_ids();
    let resolved_embedder = if embedder.eq_ignore_ascii_case("auto") {
        auto_embedder.to_string()
    } else {
        embedder.to_string()
    };
    let resolved_sparse = if sparse.eq_ignore_ascii_case("auto") {
        auto_sparse.to_string()
    } else {
        sparse.to_string()
    };
    (resolved_embedder, resolved_sparse)
}

/// Parse `nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits` output.
pub fn parse_nvidia_smi(output: &str) -> Vec<CudaGpu> {
    let mut gpus = Vec::new();
    for line in output.lines() {
        let Some((name, vram)) = line.rsplit_once(',') else {
            continue;
        };
        let name = name.trim();
        let Ok(vram_mib) = vram.trim().parse::<u64>() else {
            continue;
        };
        if !name.is_empty() && vram_mib > 0 {
            gpus.push(CudaGpu {
                name: name.to_string(),
                vram_mib,
            });
        }
    }
    gpus
}

fn probe_cuda_gpus(os: &str) -> Vec<CudaGpu> {
    let candidates: &[&str] = if os == "windows" {
        &[
            "nvidia-smi",
            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
        ]
    } else {
        &["nvidia-smi"]
    };
    for bin in candidates {
        if let Ok(output) = Command::new(bin)
            .args([
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ])
            .output()
        {
            if output.status.success() {
                let parsed = parse_nvidia_smi(&String::from_utf8_lossy(&output.stdout));
                if !parsed.is_empty() {
                    return parsed;
                }
            }
        }
    }
    Vec::new()
}

#[cfg(target_os = "macos")]
fn total_ram_bytes() -> Option<u64> {
    unsafe {
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>() as libc::size_t;
        let name = b"hw.memsize\0";
        let ok = libc::sysctlbyname(
            name.as_ptr().cast(),
            (&mut value as *mut u64).cast::<libc::c_void>(),
            &mut len,
            std::ptr::null_mut(),
            0,
        );
        (ok == 0).then_some(value)
    }
}

#[cfg(target_os = "linux")]
fn total_ram_bytes() -> Option<u64> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = text.lines().find(|l| l.starts_with("MemTotal:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb.saturating_mul(1024))
}

#[cfg(windows)]
fn total_ram_bytes() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    unsafe {
        let mut status: MEMORYSTATUSEX = std::mem::zeroed();
        status.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
        (GlobalMemoryStatusEx(&mut status) != 0).then_some(status.ullTotalPhys)
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux", windows)))]
fn total_ram_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mac(ram_gib: u64, cores: usize) -> Hardware {
        Hardware {
            os: "macos".into(),
            arch: "aarch64".into(),
            cores,
            total_ram_bytes: Some(ram_gib * GIB),
            cuda_gpus: vec![],
        }
    }

    fn win(ram_gib: u64, cores: usize, vram_mib: Option<u64>) -> Hardware {
        Hardware {
            os: "windows".into(),
            arch: "x86_64".into(),
            cores,
            total_ram_bytes: Some(ram_gib * GIB),
            cuda_gpus: vram_mib
                .map(|v| {
                    vec![CudaGpu {
                        name: "NVIDIA RTX".into(),
                        vram_mib: v,
                    }]
                })
                .unwrap_or_default(),
        }
    }

    #[test]
    fn apple_silicon_profiles_follow_vm_measured_cutoffs() {
        assert_eq!(mac(8, 8).recommend(), HardwareProfile::Bm25Only);
        assert_eq!(mac(12, 8).recommend(), HardwareProfile::Bm25Only);
        assert_eq!(mac(16, 8).recommend(), HardwareProfile::SparseAndBm25);
        assert_eq!(mac(18, 12).recommend(), HardwareProfile::SparseAndBm25);
        assert_eq!(mac(24, 8).recommend(), HardwareProfile::Full);
        assert_eq!(mac(32, 10).recommend(), HardwareProfile::Full);
        assert_eq!(mac(64, 16).recommend(), HardwareProfile::Full);
    }

    #[test]
    fn windows_profiles_gate_dense_on_cuda_and_splade_on_ram_and_cores() {
        // 8 GB Windows laptop (even with 8 cores): BM25 only.
        assert_eq!(win(8, 8, None).recommend(), HardwareProfile::Bm25Only);
        // 16 GB Windows laptop with 2 cores CPU-only: BM25 only.
        assert_eq!(win(16, 2, None).recommend(), HardwareProfile::Bm25Only);
        // 16 GB Windows laptop with 4+ cores CPU-only: SPLADE + BM25.
        assert_eq!(win(16, 4, None).recommend(), HardwareProfile::SparseAndBm25);
        // 32 GB Windows PC without CUDA GPU: SPLADE + BM25 (no 3.8 GB slow CPU dense).
        assert_eq!(
            win(32, 12, None).recommend(),
            HardwareProfile::SparseAndBm25
        );
        // 16 GB Windows PC with 6 GB VRAM card (< 8 GB): SPLADE + BM25.
        assert_eq!(
            win(16, 8, Some(6144)).recommend(),
            HardwareProfile::SparseAndBm25
        );
        // 16 GB Windows PC with RTX 4060 8 GB (8188 MiB): Full!
        assert_eq!(win(16, 8, Some(8188)).recommend(), HardwareProfile::Full);
        // 8 GB Windows PC even with GPU: host RAM < 16 GiB -> BM25 only!
        assert_eq!(win(8, 8, Some(8188)).recommend(), HardwareProfile::Bm25Only);
    }

    #[test]
    fn explicit_flags_always_override_auto_hardware_plan() {
        let low = win(8, 2, None);
        assert_eq!(
            resolve_auto_retrievers("auto", "auto", &low),
            (NONE_EMBEDDER_ID.to_string(), NONE_SPARSE_ID.to_string())
        );
        assert_eq!(
            resolve_auto_retrievers("openai:text-embedding-3-small", "auto", &low),
            (
                "openai:text-embedding-3-small".to_string(),
                NONE_SPARSE_ID.to_string()
            )
        );
        assert_eq!(
            resolve_auto_retrievers("auto", DEFAULT_SPARSE_ID, &low),
            (NONE_EMBEDDER_ID.to_string(), DEFAULT_SPARSE_ID.to_string())
        );
    }

    #[test]
    fn parse_nvidia_smi_extracts_gpu_names_and_vram() {
        let sample = "NVIDIA GeForce RTX 4060, 8188\nNVIDIA TITAN Xp, 12288\n";
        let gpus = parse_nvidia_smi(sample);
        assert_eq!(gpus.len(), 2);
        assert_eq!(gpus[0].name, "NVIDIA GeForce RTX 4060");
        assert_eq!(gpus[0].vram_mib, 8188);
        assert_eq!(gpus[1].vram_mib, 12288);
    }

    #[test]
    fn probe_runs_on_this_machine_without_failing() {
        let hw = Hardware::probe();
        assert!(hw.cores >= 1);
        assert!(hw.total_ram_bytes.is_some());
        assert!(!hw.summary().is_empty());
        assert!(!hw.rationale().is_empty());
    }
}
