//! Pure-Rust multilingual BM25 tokenization.
//!
//! Supported language presets:
//! - `multilingual` (default): script-aware segmentation that routes each
//!   contiguous script span to the appropriate analyzer (Korean -> lindera
//!   ko-dic, Japanese -> lindera IPADIC, Chinese/Han -> jieba-rs, Arabic ->
//!   light stemmer, Latin/Cyrillic/digits/identifiers -> simple lowercased
//!   alphanumeric + identifier splitting).
//! - `simple`: fast whitespace/punctuation/identifier tokenizer (Latin & code).
//! - `ko`: Korean morphological analysis (`lindera` embedded `ko-dic`, pure
//!   Rust, no external C++ shared library required) plus Latin/identifier tokens.
//! - `ja`: Japanese morphological analysis (`lindera` embedded `ipadic`).
//! - `zh`: Chinese word segmentation (`jieba-rs`).
//! - `ar`: Arabic light prefix/suffix stemmer.

use crate::error::{MinSyncError, Result};
use jieba_rs::Jieba;
use lindera::dictionary::load_dictionary;
use lindera::mode::{Mode, Penalty};
use lindera::segmenter::Segmenter;
use lindera::tokenizer::Tokenizer as LinderaTokenizer;
use std::sync::OnceLock;

pub const SUPPORTED_LANGUAGES: &[&str] = &["multilingual", "simple", "ko", "ja", "zh", "ar"];

pub const DEFAULT_LANGUAGE: &str = "multilingual";

pub fn validate_language(language: &str) -> Result<()> {
    if SUPPORTED_LANGUAGES.contains(&language) {
        Ok(())
    } else {
        Err(MinSyncError::Config(format!(
            "unsupported BM25 language {language:?}; use multilingual, simple, ko, ja, zh, or ar"
        )))
    }
}

/// Tokenize `text` according to `language` preset into normalized BM25 terms.
pub fn tokenize(text: &str, language: &str) -> Result<Vec<String>> {
    validate_language(language)?;
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    match language {
        "simple" => Ok(simple_tokens(text)),
        "ko" => korean_tokens(text),
        "ja" => japanese_tokens(text),
        "zh" => Ok(chinese_tokens(text)),
        "ar" => Ok(arabic_tokens(text)),
        "multilingual" => multilingual_tokens(text),
        _ => unreachable!(),
    }
}

/// Tokenize `text` and return a space-joined string (for debugging / display).
pub fn tokenize_to_string(text: &str, language: &str) -> Result<String> {
    Ok(tokenize(text, language)?.join(" "))
}

/// Lowercased tokens from whitespace/punctuation splitting, also preserving
/// exact identifiers (e.g. `ERR_404`, `foo-bar`, `v1.2.3`) and splitting
/// compound code identifiers on `_`/`-`/`.` so both exact and sub-token
/// queries match.
fn simple_tokens(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split(|c: char| {
        c.is_whitespace() || (c.is_ascii_punctuation() && c != '_' && c != '-' && c != '.')
    }) {
        let trimmed = raw.trim_matches(|c: char| !c.is_alphanumeric());
        if trimmed.is_empty() {
            continue;
        }
        let lower = trimmed.to_lowercase();
        let has_sep = lower.contains(['_', '-', '.']);
        out.push(lower.clone());
        if has_sep {
            for part in lower.split(['_', '-', '.']) {
                if !part.is_empty() && part != lower {
                    out.push(part.to_string());
                }
            }
        }
    }
    out
}

fn ko_tokenizer() -> Result<&'static LinderaTokenizer> {
    static TOKENIZER: OnceLock<std::result::Result<LinderaTokenizer, String>> = OnceLock::new();
    TOKENIZER
        .get_or_init(|| {
            load_dictionary("embedded://ko-dic")
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
        .map_err(|error| MinSyncError::Config(format!("initialize Korean tokenizer: {error}")))
}

fn ja_tokenizer() -> Result<&'static LinderaTokenizer> {
    static TOKENIZER: OnceLock<std::result::Result<LinderaTokenizer, String>> = OnceLock::new();
    TOKENIZER
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
        .map_err(|error| MinSyncError::Config(format!("initialize Japanese tokenizer: {error}")))
}

fn zh_tokenizer() -> &'static Jieba {
    static TOKENIZER: OnceLock<Jieba> = OnceLock::new();
    TOKENIZER.get_or_init(Jieba::new)
}

/// Filter out pure punctuation/whitespace tokens produced by morphological
/// analyzers while keeping all alphanumeric/ideographic/syllabic terms.
fn is_meaningful_token(s: &str) -> bool {
    !s.is_empty() && s.chars().any(|c| c.is_alphanumeric())
}

fn lindera_tokenize(tokenizer: &LinderaTokenizer, text: &str, label: &str) -> Result<Vec<String>> {
    let tokens = tokenizer
        .tokenize(text)
        .map_err(|error| MinSyncError::Config(format!("tokenize {label} text: {error}")))?;
    let mut out = Vec::with_capacity(tokens.len());
    for token in tokens {
        let surface = token.surface.trim();
        if is_meaningful_token(surface) {
            out.push(surface.to_lowercase());
        }
    }
    Ok(out)
}

fn korean_tokens(text: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut non_hangul = String::new();
    let mut hangul = String::new();

    for ch in text.chars() {
        if is_hangul(ch) {
            if !non_hangul.is_empty() {
                out.extend(simple_tokens(&non_hangul));
                non_hangul.clear();
            }
            hangul.push(ch);
        } else {
            if !hangul.is_empty() {
                out.extend(lindera_tokenize(ko_tokenizer()?, &hangul, "Korean")?);
                hangul.clear();
            }
            non_hangul.push(ch);
        }
    }
    if !hangul.is_empty() {
        out.extend(lindera_tokenize(ko_tokenizer()?, &hangul, "Korean")?);
    }
    if !non_hangul.is_empty() {
        out.extend(simple_tokens(&non_hangul));
    }
    Ok(out)
}

fn japanese_tokens(text: &str) -> Result<Vec<String>> {
    lindera_tokenize(ja_tokenizer()?, text, "Japanese")
}

fn chinese_tokens(text: &str) -> Vec<String> {
    let jieba = zh_tokenizer();
    jieba
        .cut_for_search(text, true)
        .into_iter()
        .map(|token| token.word.trim())
        .filter(|s| is_meaningful_token(s))
        .map(str::to_lowercase)
        .collect()
}

fn arabic_tokens(text: &str) -> Vec<String> {
    const PREFIXES: &[&str] = &[
        "وال", "فال", "بال", "كال", "لال", "ال", "لل", "و", "ف", "ب", "ك", "ل",
    ];
    const SUFFIXES: &[&str] = &["ها", "ان", "ات", "ون", "ين", "يه", "ية", "ه", "ة", "ي"];
    let mut tokens = Vec::new();
    for word in text.split(|c: char| !c.is_alphanumeric()) {
        if word.is_empty() {
            continue;
        }
        let word = word.to_lowercase();
        tokens.push(word.clone());
        if word.chars().any(is_arabic) {
            let mut stem = word.clone();
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
            if stem != word {
                tokens.push(stem);
            }
        }
    }
    tokens
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Script {
    Hangul,
    Kana,
    Han,
    Arabic,
    Other,
}

fn is_hangul(c: char) -> bool {
    matches!(
        c as u32,
        0xAC00..=0xD7AF // Hangul Syllables
        | 0x1100..=0x11FF // Hangul Jamo
        | 0x3130..=0x318F // Hangul Compatibility Jamo
        | 0xA960..=0xA97F // Hangul Jamo Extended-A
        | 0xD7B0..=0xD7FF // Hangul Jamo Extended-B
    )
}

fn is_kana(c: char) -> bool {
    matches!(
        c as u32,
        0x3040..=0x309F // Hiragana
        | 0x30A0..=0x30FF // Katakana
        | 0x31F0..=0x31FF // Katakana Phonetic Extensions
        | 0xFF66..=0xFF9F // Halfwidth Katakana
    )
}

fn is_han(c: char) -> bool {
    matches!(
        c as u32,
        0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0x3400..=0x4DBF // CJK Extension A
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0x20000..=0x2A6DF // CJK Extension B
    )
}

fn is_arabic(c: char) -> bool {
    matches!(
        c as u32,
        0x0600..=0x06FF
        | 0x0750..=0x077F
        | 0x08A0..=0x08FF
        | 0xFB50..=0xFDFF
        | 0xFE70..=0xFEFF
    )
}

fn classify_char(c: char) -> Script {
    if is_hangul(c) {
        Script::Hangul
    } else if is_kana(c) {
        Script::Kana
    } else if is_han(c) {
        Script::Han
    } else if is_arabic(c) {
        Script::Arabic
    } else {
        Script::Other
    }
}

/// Script-aware multilingual tokenization:
/// - Hangul spans -> `lindera` ko-dic (Korean morphological decomposition)
/// - Kana spans -> `lindera` IPADIC (Japanese morphological decomposition)
/// - Han spans -> BOTH `jieba-rs` (Chinese segmentation) AND `lindera` IPADIC
///   (Japanese Kanji compounds), merged with per-span dedup. This makes
///   `"我们中出了一个叛徒"` (Chinese) and `"関西国際空港"` (pure-Kanji Japanese)
///   both match reliably without fragile document-level script heuristics.
/// - Arabic spans -> Arabic light stemmer
/// - Latin, Cyrillic, numbers, code identifiers -> `simple_tokens`
fn multilingual_tokens(text: &str) -> Result<Vec<String>> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut current_script: Option<Script> = None;

    let flush = |script: Script, buf: &str, out: &mut Vec<String>| -> Result<()> {
        if buf.is_empty() {
            return Ok(());
        }
        match script {
            Script::Hangul => {
                out.extend(lindera_tokenize(ko_tokenizer()?, buf, "Korean")?);
            }
            Script::Kana => {
                out.extend(lindera_tokenize(ja_tokenizer()?, buf, "Japanese")?);
            }
            Script::Han => {
                // Pure Han span (Chinese or pure-Kanji Japanese): run both
                // analyzers and union their distinct tokens so Chinese text
                // and pure-Kanji Japanese queries both match.
                let zh = chinese_tokens(buf);
                let ja = lindera_tokenize(ja_tokenizer()?, buf, "Japanese")?;
                if zh == ja {
                    out.extend(zh);
                } else {
                    out.extend(zh.iter().cloned());
                    for token in ja {
                        if !zh.contains(&token) {
                            out.push(token);
                        }
                    }
                }
            }
            Script::Arabic => {
                out.extend(arabic_tokens(buf));
            }
            Script::Other => {
                out.extend(simple_tokens(buf));
            }
        }
        Ok(())
    };

    for ch in text.chars() {
        let script = classify_char(ch);
        match current_script {
            Some(prev) if prev == script => current.push(ch),
            Some(prev) => {
                flush(prev, &current, &mut out)?;
                current.clear();
                current.push(ch);
                current_script = Some(script);
            }
            None => {
                current.push(ch);
                current_script = Some(script);
            }
        }
    }
    if let Some(prev) = current_script {
        flush(prev, &current, &mut out)?;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_validation_accepts_known_and_rejects_unknown() {
        for lang in SUPPORTED_LANGUAGES {
            assert!(validate_language(lang).is_ok(), "{lang} should be valid");
        }
        assert!(validate_language("klingon").is_err());
    }

    #[test]
    fn korean_splits_compound_and_particles() {
        let tokens = tokenize("아버지가방에들어가신다", "ko").expect("ko tokenize");
        assert!(
            tokens.contains(&"아버지".to_string()),
            "expected 아버지 in {tokens:?}"
        );
        assert!(
            tokens.contains(&"방".to_string()),
            "expected 방 in {tokens:?}"
        );

        let refund = tokenize("환불정책과배송안내입니다", "ko").expect("ko tokenize");
        assert!(refund.contains(&"환불".to_string()), "{refund:?}");
        assert!(refund.contains(&"정책".to_string()), "{refund:?}");
        assert!(refund.contains(&"배송".to_string()), "{refund:?}");
        assert!(refund.contains(&"안내".to_string()), "{refund:?}");
    }

    #[test]
    fn japanese_splits_compounds_and_kana() {
        let tokens = tokenize("関西国際空港限定トートバッグ", "ja").expect("ja tokenize");
        assert!(tokens.contains(&"関西".to_string()), "{tokens:?}");
        assert!(tokens.contains(&"空港".to_string()), "{tokens:?}");
    }

    #[test]
    fn chinese_splits_sentences() {
        let tokens = tokenize("我们中出了一个叛徒", "zh").expect("zh tokenize");
        assert!(tokens.contains(&"我们".to_string()), "{tokens:?}");
        assert!(tokens.contains(&"叛徒".to_string()), "{tokens:?}");
    }

    #[test]
    fn arabic_stems_prefixes_and_suffixes() {
        let tokens = tokenize("والكتاب في المدرسة", "ar").expect("ar tokenize");
        assert!(tokens.contains(&"كتاب".to_string()), "{tokens:?}");
    }

    #[test]
    fn simple_preserves_and_splits_identifiers() {
        let tokens = tokenize("Error ERR_404 in user-service.v2", "simple").expect("simple");
        assert!(tokens.contains(&"err_404".to_string()), "{tokens:?}");
        assert!(tokens.contains(&"err".to_string()), "{tokens:?}");
        assert!(tokens.contains(&"404".to_string()), "{tokens:?}");
        assert!(
            tokens.contains(&"user-service.v2".to_string()),
            "{tokens:?}"
        );
        assert!(tokens.contains(&"service".to_string()), "{tokens:?}");
    }

    #[test]
    fn multilingual_handles_mixed_en_ko_ja_zh_ar_in_one_document() {
        let text = "MinSync ERR_404 환불정책과배송안내 関西国際空港限定トートバッグ 我们中出了一个叛徒 والكتاب";
        let tokens = tokenize(text, "multilingual").expect("multilingual tokenize");
        for expected in [
            "minsync", "err_404", "환불", "정책", "배송", "関西", "空港", "我们", "叛徒", "كتاب",
        ] {
            assert!(
                tokens.contains(&expected.to_string()),
                "missing {expected:?} in {tokens:?}"
            );
        }
    }

    #[test]
    fn multilingual_pure_kanji_query_matches_kana_kanji_japanese_doc() {
        let doc_tokens = tokenize("関西国際空港で限定品を買う", "multilingual").unwrap();
        let query_tokens = tokenize("関西国際空港", "multilingual").unwrap();
        assert!(
            query_tokens.iter().any(|t| doc_tokens.contains(t)),
            "query {query_tokens:?} should overlap doc {doc_tokens:?}"
        );
    }
}
