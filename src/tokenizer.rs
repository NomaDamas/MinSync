use crate::error::{MinSyncError, Result};
use jieba_rs::Jieba;
use kiwi_rs::Kiwi;
use lindera::dictionary::load_dictionary;
use lindera::mode::Mode;
use lindera::mode::Penalty;
use lindera::segmenter::Segmenter;
use lindera::tokenizer::Tokenizer as LinderaTokenizer;
use std::cell::RefCell;
use std::ffi::CString;
use std::sync::{Mutex, OnceLock};

pub const SUPPORTED_LANGUAGES: &[&str] = &["simple", "ko", "ja", "zh", "ar", "multilingual"];

pub fn validate_language(language: &str) -> Result<()> {
    if SUPPORTED_LANGUAGES.contains(&language) {
        Ok(())
    } else {
        Err(MinSyncError::Config(format!(
            "unsupported BM25 language {language:?}; use simple, ko, ja, zh, ar, or multilingual"
        )))
    }
}

pub fn tokenize(text: &str, language: &str) -> Result<String> {
    validate_language(language)?;
    match language {
        "simple" => Ok(simple_tokens(text)),
        "ko" => korean_tokens(text),
        "ja" => japanese_tokens(text),
        "zh" => Ok(chinese_tokens(text)),
        "ar" => Ok(arabic_tokens(text)),
        "multilingual" => Ok([
            simple_tokens(text),
            korean_tokens(text)?,
            japanese_tokens(text)?,
            chinese_tokens(text),
            arabic_tokens(text),
        ]
        .join(" ")),
        _ => unreachable!(),
    }
}

fn simple_tokens(text: &str) -> String {
    text.split_whitespace()
        .map(|token| token.trim_matches(|c: char| c.is_ascii_punctuation()))
        .filter(|token| !token.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

fn korean_tokens(text: &str) -> Result<String> {
    Ok(korean_analysis(text)?
        .into_iter()
        .map(|(form, _)| form)
        .collect::<Vec<_>>()
        .join(" "))
}

fn korean_analysis(text: &str) -> Result<Vec<(String, String)>> {
    with_native_stderr_suppressed(|| korean_analysis_inner(text))
}

#[cfg(windows)]
fn korean_analysis_inner(text: &str) -> Result<Vec<(String, String)>> {
    thread_local! {
        // Kiwi's Windows native destructor can raise STATUS_ACCESS_VIOLATION
        // during thread-local teardown. Keep one instance alive until process
        // exit; the bounded leak avoids unloading native state prematurely.
        static TOKENIZER: RefCell<Option<&'static Kiwi>> = const { RefCell::new(None) };
    }
    TOKENIZER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            let kiwi = Kiwi::init().map_err(|error| {
                MinSyncError::Config(format!("initialize Korean tokenizer: {error}"))
            })?;
            *slot = Some(Box::leak(Box::new(kiwi)));
        }
        slot.as_ref()
            .expect("Kiwi initialized above")
            .tokenize(text)
            .map(|tokens| {
                tokens
                    .into_iter()
                    .map(|token| (token.form, token.tag))
                    .collect()
            })
            .map_err(|error| MinSyncError::Config(format!("tokenize Korean text: {error}")))
    })
}

#[cfg(not(windows))]
fn korean_analysis_inner(text: &str) -> Result<Vec<(String, String)>> {
    thread_local! {
        static TOKENIZER: RefCell<Option<Kiwi>> = const { RefCell::new(None) };
    }
    TOKENIZER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_none() {
            *slot = Some(Kiwi::init().map_err(|error| {
                MinSyncError::Config(format!("initialize Korean tokenizer: {error}"))
            })?);
        }
        slot.as_ref()
            .expect("Kiwi initialized above")
            .tokenize(text)
            .map(|tokens| {
                tokens
                    .into_iter()
                    .map(|token| (token.form, token.tag))
                    .collect()
            })
            .map_err(|error| MinSyncError::Config(format!("tokenize Korean text: {error}")))
    })
}

fn with_native_stderr_suppressed<T>(op: impl FnOnce() -> T) -> T {
    static LOCK: Mutex<()> = Mutex::new(());
    let _lock = LOCK.lock().unwrap_or_else(|error| error.into_inner());
    let Some(guard) = StderrRestore::silence() else {
        return op();
    };
    let result = op();
    drop(guard);
    result
}

struct StderrRestore {
    saved: i32,
}

impl StderrRestore {
    fn silence() -> Option<Self> {
        const STDERR_FD: i32 = 2;
        // SAFETY: dup copies the current stderr fd so Drop can restore it.
        let saved = unsafe { libc::dup(STDERR_FD) };
        if saved < 0 {
            return None;
        }
        let null_path = if cfg!(windows) { "NUL" } else { "/dev/null" };
        let Ok(path) = CString::new(null_path) else {
            // SAFETY: saved is a valid duplicated fd from dup().
            unsafe { libc::close(saved) };
            return None;
        };
        // SAFETY: path is a valid C string for the platform discard device.
        let null_fd = unsafe { libc::open(path.as_ptr(), libc::O_WRONLY) };
        if null_fd < 0 {
            // SAFETY: saved is still the duplicated stderr fd.
            unsafe { libc::close(saved) };
            return None;
        }
        unsafe {
            // SAFETY: null_fd is an open discard fd; STDERR_FILENO is process stderr.
            libc::dup2(null_fd, STDERR_FD);
            libc::close(null_fd);
        }
        Some(Self { saved })
    }
}

impl Drop for StderrRestore {
    fn drop(&mut self) {
        const STDERR_FD: i32 = 2;
        unsafe {
            // SAFETY: saved is the original stderr fd duplicated in silence().
            libc::dup2(self.saved, STDERR_FD);
            libc::close(self.saved);
        }
    }
}

fn japanese_tokens(text: &str) -> Result<String> {
    static TOKENIZER: OnceLock<std::result::Result<LinderaTokenizer, String>> = OnceLock::new();
    let tokenizer = TOKENIZER
        .get_or_init(|| {
            load_dictionary("embedded://ipadic")
                .map(|dictionary| {
                    LinderaTokenizer::new(Segmenter::new(
                        Mode::Decompose(Penalty::default()),
                        dictionary,
                        None,
                    ))
                })
                .map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|error| MinSyncError::Config(format!("initialize Japanese tokenizer: {error}")))?;
    let tokens = tokenizer
        .tokenize(text)
        .map_err(|error| MinSyncError::Config(format!("tokenize Japanese text: {error}")))?;
    Ok(tokens
        .into_iter()
        .map(|token| token.surface.to_string())
        .collect::<Vec<_>>()
        .join(" "))
}

fn chinese_tokens(text: &str) -> String {
    static TOKENIZER: OnceLock<Jieba> = OnceLock::new();
    TOKENIZER.get_or_init(Jieba::new).cut(text, false).join(" ")
}

fn arabic_tokens(text: &str) -> String {
    const PREFIXES: &[&str] = &[
        "وال", "فال", "بال", "كال", "لال", "ال", "لل", "و", "ف", "ب", "ك", "ل",
    ];
    const SUFFIXES: &[&str] = &["ها", "ان", "ات", "ون", "ين", "يه", "ية", "ه", "ة", "ي"];
    let mut tokens = Vec::new();
    for word in text.split(|c: char| !c.is_alphabetic()) {
        if word.is_empty() {
            continue;
        }
        let word = word.to_lowercase();
        tokens.push(word.clone());
        let mut stem = word;
        for prefix in PREFIXES {
            if let Some(rest) = stem.strip_prefix(prefix) {
                if rest.chars().count() >= 2 {
                    stem = rest.to_string();
                    break;
                }
            }
        }
        for suffix in SUFFIXES {
            if stem.ends_with(suffix) && stem.chars().count() > suffix.chars().count() + 1 {
                stem.truncate(stem.len() - suffix.len());
                break;
            }
        }
        if !tokens.contains(&stem) {
            tokens.push(stem);
        }
    }
    tokens.join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multilingual_presets_split_language_specific_words() {
        assert!(tokenize("아버지가방에들어가신다", "ko")
            .expect("Korean tokenize")
            .contains("아버지"));
        assert!(tokenize("関西国際空港限定トートバッグ", "ja")
            .expect("Japanese tokenize")
            .contains("関西"));
        assert!(tokenize("我们中出了一个叛徒", "zh")
            .expect("Chinese tokenize")
            .contains("我们"));
        assert!(tokenize("والكتاب في المدرسة", "ar")
            .expect("Arabic tokenize")
            .contains("كتاب"));
    }

    #[test]
    fn language_validation_rejects_unknown_presets() {
        assert!(validate_language("klingon").is_err());
    }

    #[test]
    fn kiwi_matches_kiwipiepy_reference_forms_and_tags() {
        let cases = [
            (
                "아버지가방에들어가신다",
                vec![
                    ("아버지", "NNG"),
                    ("가", "JKS"),
                    ("방", "NNG"),
                    ("에", "JKB"),
                    ("들어가", "VV"),
                    ("시", "EP"),
                    ("ᆫ다", "EF"),
                ],
            ),
            (
                "오늘저녁먹음",
                vec![("오늘", "NNG"), ("저녁", "NNG"), ("먹", "VV"), ("음", "EF")],
            ),
            (
                "서울맛집추천",
                vec![("서울", "NNP"), ("맛집", "NNG"), ("추천", "NNG")],
            ),
            (
                "환불정책과배송안내입니다",
                vec![
                    ("환불", "NNG"),
                    ("정책", "NNG"),
                    ("과", "JC"),
                    ("배송", "NNG"),
                    ("안내", "NNG"),
                    ("이", "VCP"),
                    ("ᆸ니다", "EF"),
                ],
            ),
        ];
        for (text, expected) in cases {
            let actual = korean_analysis(text).expect("Kiwi analysis succeeds");
            assert_eq!(
                actual,
                expected
                    .into_iter()
                    .map(|(form, tag)| (form.to_string(), tag.to_string()))
                    .collect::<Vec<_>>()
            );
        }
    }
}
