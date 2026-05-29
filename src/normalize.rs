use crate::config::NormalizeConfig;

pub fn normalize_text(text: &str, config: &NormalizeConfig) -> String {
    let mut result = text.to_string();

    if config.normalize_newlines {
        result = result.replace("\r\n", "\n").replace('\r', "\n");
    }

    if config.strip_frontmatter {
        result = strip_yaml_frontmatter(&result);
    }

    if config.strip_trailing_whitespace {
        result = result
            .lines()
            .map(str::trim_end)
            .collect::<Vec<_>>()
            .join("\n");
    }

    if config.collapse_whitespace {
        result = collapse_spaces_and_tabs(&result);
    }

    result
}

fn strip_yaml_frontmatter(text: &str) -> String {
    let normalized = text.replace("\r\n", "\n");
    if !normalized.starts_with("---\n") {
        return text.to_string();
    }

    let Some(end_pos) = normalized[4..].find("\n---") else {
        return text.to_string();
    };

    let closing_start = 4 + end_pos + 1;
    let closing_line_end = normalized[closing_start..]
        .find('\n')
        .map(|offset| closing_start + offset + 1)
        .unwrap_or(normalized.len());

    normalized[closing_line_end..]
        .strip_prefix('\n')
        .unwrap_or(&normalized[closing_line_end..])
        .to_string()
}

fn collapse_spaces_and_tabs(text: &str) -> String {
    let mut collapsed = String::with_capacity(text.len());
    let mut prev_was_space = false;

    for ch in text.chars() {
        if ch == ' ' || ch == '\t' {
            if !prev_was_space {
                collapsed.push(' ');
            }
            prev_was_space = true;
        } else {
            prev_was_space = false;
            collapsed.push(ch);
        }
    }

    collapsed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::NormalizeConfig;

    #[test]
    fn test_normalize_newlines() {
        let config = NormalizeConfig {
            normalize_newlines: true,
            strip_trailing_whitespace: false,
            collapse_whitespace: false,
            strip_frontmatter: false,
        };

        assert_eq!(
            normalize_text("hello\r\nworld\ragain", &config),
            "hello\nworld\nagain"
        );
    }

    #[test]
    fn test_strip_trailing_whitespace() {
        let config = NormalizeConfig {
            strip_trailing_whitespace: true,
            normalize_newlines: false,
            collapse_whitespace: false,
            strip_frontmatter: false,
        };

        assert_eq!(normalize_text("hello   \nworld  ", &config), "hello\nworld");
    }

    #[test]
    fn test_strip_frontmatter() {
        let config = NormalizeConfig {
            strip_frontmatter: true,
            normalize_newlines: true,
            strip_trailing_whitespace: false,
            collapse_whitespace: false,
        };

        assert_eq!(
            normalize_text("---\ntitle: x\n---\ncontent", &config),
            "content"
        );
    }

    #[test]
    fn test_collapse_whitespace() {
        let config = NormalizeConfig {
            collapse_whitespace: true,
            normalize_newlines: false,
            strip_trailing_whitespace: false,
            strip_frontmatter: false,
        };

        assert_eq!(normalize_text("hello   world", &config), "hello world");
    }

    #[test]
    fn test_normalize_defaults() {
        assert_eq!(
            normalize_text("hello  \r\nworld\t ", &NormalizeConfig::default()),
            "hello\nworld"
        );
    }

    #[test]
    fn test_normalize_no_ops() {
        let config = NormalizeConfig {
            strip_trailing_whitespace: false,
            normalize_newlines: false,
            collapse_whitespace: false,
            strip_frontmatter: false,
        };
        let text = "---\r\ntitle: x\r\n---\r\nhello   \tworld  ";

        assert_eq!(normalize_text(text, &config), text);
    }
}
