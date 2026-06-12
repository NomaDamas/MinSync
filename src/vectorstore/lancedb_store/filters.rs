//! Translation of [`Filter`] trees into LanceDB SQL predicates, including
//! field whitelisting and literal escaping.

use crate::error::{MinSyncError, Result};
use crate::vectorstore::Filter;

pub(super) const FILTERABLE_FIELDS: &[&str] = &[
    "id",
    "text",
    "source_id",
    "path",
    "chunk_schema_id",
    "chunk_type",
    "heading_path",
    "content_hash",
    "seen_token",
];

pub fn filter_to_sql(filter: &Filter) -> Result<String> {
    match filter {
        Filter::Eq(field, value) => Ok(format!(
            "{} = '{}'",
            validate_filter_field(field)?,
            escape_sql_literal(value)
        )),
        Filter::Neq(field, value) => Ok(format!(
            "{} != '{}'",
            validate_filter_field(field)?,
            escape_sql_literal(value)
        )),
        Filter::And(filters) if filters.is_empty() => Ok("TRUE".to_string()),
        Filter::And(filters) => filters
            .iter()
            .map(|nested| filter_to_sql(nested).map(|sql| format!("({sql})")))
            .collect::<Result<Vec<_>>>()
            .map(|parts| parts.join(" AND ")),
    }
}

fn validate_filter_field(field: &str) -> Result<&str> {
    if FILTERABLE_FIELDS.contains(&field) {
        Ok(field)
    } else {
        Err(MinSyncError::VectorStore(format!(
            "unsupported LanceDB filter field: {field}"
        )))
    }
}

pub(super) fn id_in_filter<'a>(ids: impl Iterator<Item = &'a str>) -> String {
    let values = ids
        .map(|id| format!("'{}'", escape_sql_literal(id)))
        .collect::<Vec<_>>()
        .join(", ");
    format!("id IN ({values})")
}

pub(super) fn sql_literal(value: &str) -> String {
    format!("'{}'", escape_sql_literal(value))
}

pub(super) fn escape_sql_literal(value: &str) -> String {
    value.replace('\'', "''")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_to_sql() {
        assert_eq!(
            filter_to_sql(&Filter::Eq("path".to_string(), "a'b".to_string())).expect("sql"),
            "path = 'a''b'"
        );
        assert_eq!(
            filter_to_sql(&Filter::Neq("seen_token".to_string(), "old".to_string())).expect("sql"),
            "seen_token != 'old'"
        );
        assert_eq!(
            filter_to_sql(&Filter::And(vec![
                Filter::Eq("path".to_string(), "x.txt".to_string()),
                Filter::Neq("seen_token".to_string(), "keep".to_string()),
            ]))
            .expect("sql"),
            "(path = 'x.txt') AND (seen_token != 'keep')"
        );
        assert_eq!(filter_to_sql(&Filter::And(vec![])).expect("sql"), "TRUE");
        assert!(filter_to_sql(&Filter::Eq("bad".to_string(), "x".to_string())).is_err());
    }

    #[test]
    fn test_id_in_filter_escapes_literals() {
        assert_eq!(
            id_in_filter(["a", "o'brien"].into_iter()),
            "id IN ('a', 'o''brien')"
        );
        assert_eq!(id_in_filter(std::iter::empty()), "id IN ()");
    }

    #[test]
    fn test_sql_literal_quotes_and_escapes() {
        assert_eq!(sql_literal("plain"), "'plain'");
        assert_eq!(sql_literal("it's"), "'it''s'");
    }

    #[test]
    fn test_nested_and_filters_parenthesized() {
        let sql = filter_to_sql(&Filter::And(vec![
            Filter::Eq("path".to_string(), "x".to_string()),
            Filter::And(vec![
                Filter::Eq("source_id".to_string(), "s".to_string()),
                Filter::Neq("seen_token".to_string(), "t".to_string()),
            ]),
        ]))
        .expect("sql");
        assert_eq!(
            sql,
            "(path = 'x') AND ((source_id = 's') AND (seen_token != 't'))"
        );
    }
}
