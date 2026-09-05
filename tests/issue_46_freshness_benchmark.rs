use minsync::manifest::Manifest;
use std::time::{Duration, Instant};

const CORPUS_SIZES: [usize; 4] = [64, 256, 512, 1024];
const FILE_BYTES: usize = 16 * 1024;
const SAMPLES: usize = 15;

#[derive(Debug)]
struct Measurement {
    p50: Duration,
    p95: Duration,
    files_examined: usize,
    files_rehashed: usize,
    bytes_hashed: u64,
}

fn percentile(samples: &mut [Duration], percentile: usize) -> Duration {
    samples.sort_unstable();
    let index = (samples.len() - 1) * percentile / 100;
    samples[index]
}

fn measure(root: &std::path::Path, baseline: Option<&Manifest>) -> Measurement {
    let mut durations = Vec::with_capacity(SAMPLES);
    let mut final_stats = None;

    for _ in 0..SAMPLES {
        let started = Instant::now();
        let (_manifest, stats) = Manifest::scan_with_baseline_stats(root, "benchmark", baseline)
            .expect("benchmark scan succeeds");
        durations.push(started.elapsed());
        final_stats = Some(stats);
    }

    let stats = final_stats.expect("at least one sample");
    let mut p50_samples = durations.clone();
    Measurement {
        p50: percentile(&mut p50_samples, 50),
        p95: percentile(&mut durations, 95),
        files_examined: stats.files_examined,
        files_rehashed: stats.files_rehashed,
        bytes_hashed: stats.bytes_hashed,
    }
}

#[test]
#[ignore = "deterministic performance measurement; run with --ignored --nocapture"]
fn benchmark_issue_46_unchanged_freshness() {
    println!(
        "corpus_files,file_bytes,mode,p50_us,p95_us,files_examined,files_rehashed,bytes_hashed"
    );

    for file_count in CORPUS_SIZES {
        let dir = tempfile::tempdir().expect("create benchmark tempdir");
        let content = vec![b'x'; FILE_BYTES];
        for index in 0..file_count {
            std::fs::write(dir.path().join(format!("file-{index:04}.txt")), &content)
                .expect("write benchmark file");
        }

        let baseline = Manifest::scan(dir.path(), "benchmark").expect("create baseline");
        let current = measure(dir.path(), None);
        let optimized = measure(dir.path(), Some(&baseline));

        for (mode, measurement) in [("current_full_hash", current), ("optimized", optimized)] {
            println!(
                "{file_count},{FILE_BYTES},{mode},{},{},{},{},{}",
                measurement.p50.as_micros(),
                measurement.p95.as_micros(),
                measurement.files_examined,
                measurement.files_rehashed,
                measurement.bytes_hashed
            );
        }
    }
}
