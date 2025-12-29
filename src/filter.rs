use serde_json::Value;

/// Advanced metadata filter for search results
#[derive(Debug, Clone)]
pub enum Filter {
    /// Match exact value: {"key": "value"}
    Equals { key: String, value: Value },

    /// Not equal: {"key": {"$ne": "value"}}
    NotEquals { key: String, value: Value },

    /// Greater than: {"key": {"$gt": 5}}
    GreaterThan { key: String, value: f64 },

    /// Less than: {"key": {"$lt": 5}}
    LessThan { key: String, value: f64 },

    /// Greater than or equal: {"key": {"$gte": 5}}
    GreaterOrEqual { key: String, value: f64 },

    /// Less than or equal: {"key": {"$lte": 5}}
    LessOrEqual { key: String, value: f64 },

    /// Between (inclusive): {"key": {"$between": [min, max]}}
    Between { key: String, min: f64, max: f64 },

    /// In array: {"key": {"$in": [1, 2, 3]}}
    In { key: String, values: Vec<Value> },

    /// Not in array: {"key": {"$nin": [1, 2, 3]}}
    NotIn { key: String, values: Vec<Value> },

    /// Exists: {"key": {"$exists": true}}
    Exists { key: String, exists: bool },

    /// String starts with: {"key": {"$startsWith": "prefix"}}
    StartsWith { key: String, prefix: String },

    /// String ends with: {"key": {"$endsWith": "suffix"}}
    EndsWith { key: String, suffix: String },

    /// String contains: {"key": {"$contains": "substring"}}
    Contains { key: String, substring: String },

    /// Regex match: {"key": {"$regex": "pattern"}}
    Regex { key: String, pattern: String },

    /// AND multiple conditions
    And(Vec<Filter>),

    /// OR multiple conditions
    Or(Vec<Filter>),

    /// NOT (negation): {"$not": {...}}
    #[allow(dead_code)]
    Not(Box<Filter>),
}

impl Filter {
    /// Evaluate filter against metadata JSON
    pub fn matches(&self, metadata: &Value) -> bool {
        match self {
            Filter::Equals { key, value } => get_nested_value(metadata, key)
                .map(|v| v == value)
                .unwrap_or(false),

            Filter::NotEquals { key, value } => get_nested_value(metadata, key)
                .map(|v| v != value)
                .unwrap_or(true),

            Filter::GreaterThan { key, value } => get_nested_value(metadata, key)
                .and_then(|v| v.as_f64())
                .map(|v| v > *value)
                .unwrap_or(false),

            Filter::LessThan { key, value } => get_nested_value(metadata, key)
                .and_then(|v| v.as_f64())
                .map(|v| v < *value)
                .unwrap_or(false),

            Filter::GreaterOrEqual { key, value } => get_nested_value(metadata, key)
                .and_then(|v| v.as_f64())
                .map(|v| v >= *value)
                .unwrap_or(false),

            Filter::LessOrEqual { key, value } => get_nested_value(metadata, key)
                .and_then(|v| v.as_f64())
                .map(|v| v <= *value)
                .unwrap_or(false),

            Filter::Between { key, min, max } => get_nested_value(metadata, key)
                .and_then(|v| v.as_f64())
                .map(|v| v >= *min && v <= *max)
                .unwrap_or(false),

            Filter::In { key, values } => get_nested_value(metadata, key)
                .map(|v| values.contains(v))
                .unwrap_or(false),

            Filter::NotIn { key, values } => get_nested_value(metadata, key)
                .map(|v| !values.contains(v))
                .unwrap_or(true),

            Filter::Exists { key, exists } => {
                let found = get_nested_value(metadata, key).is_some();
                found == *exists
            },

            Filter::StartsWith { key, prefix } => get_nested_value(metadata, key)
                .and_then(|v| v.as_str())
                .map(|s| s.starts_with(prefix))
                .unwrap_or(false),

            Filter::EndsWith { key, suffix } => get_nested_value(metadata, key)
                .and_then(|v| v.as_str())
                .map(|s| s.ends_with(suffix))
                .unwrap_or(false),

            Filter::Contains { key, substring } => get_nested_value(metadata, key)
                .and_then(|v| v.as_str())
                .map(|s| s.contains(substring.as_str()))
                .unwrap_or(false),

            Filter::Regex { key, pattern } => {
                if let Ok(re) = regex::Regex::new(pattern) {
                    get_nested_value(metadata, key)
                        .and_then(|v| v.as_str())
                        .map(|s| re.is_match(s))
                        .unwrap_or(false)
                } else {
                    false
                }
            },

            Filter::And(filters) => filters.iter().all(|f| f.matches(metadata)),

            Filter::Or(filters) => filters.iter().any(|f| f.matches(metadata)),

            Filter::Not(filter) => !filter.matches(metadata),
        }
    }
}

/// Get nested value from JSON using dot notation
/// Example: "user.name" -> metadata["user"]["name"]
fn get_nested_value<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    let parts: Vec<&str> = key.split('.').collect();
    let mut current = value;

    for part in parts {
        match current {
            Value::Object(map) => {
                current = map.get(part)?;
            },
            _ => return None,
        }
    }

    Some(current)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_equals_filter() {
        let metadata = json!({
            "category": "tech",
            "score": 0.8
        });

        let filter = Filter::Equals {
            key: "category".to_string(),
            value: json!("tech"),
        };

        assert!(filter.matches(&metadata));

        let filter2 = Filter::Equals {
            key: "category".to_string(),
            value: json!("science"),
        };

        assert!(!filter2.matches(&metadata));
    }

    #[test]
    fn test_numeric_filters() {
        let metadata = json!({
            "score": 0.8
        });

        assert!(Filter::GreaterThan {
            key: "score".to_string(),
            value: 0.5,
        }
        .matches(&metadata));

        assert!(!Filter::GreaterThan {
            key: "score".to_string(),
            value: 0.9,
        }
        .matches(&metadata));

        assert!(Filter::LessThan {
            key: "score".to_string(),
            value: 0.9,
        }
        .matches(&metadata));
    }

    #[test]
    fn test_in_filter() {
        let metadata = json!({
            "category": "tech"
        });

        let filter = Filter::In {
            key: "category".to_string(),
            values: vec![json!("tech"), json!("science")],
        };

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_and_filter() {
        let metadata = json!({
            "category": "tech",
            "score": 0.8
        });

        let filter = Filter::And(vec![
            Filter::Equals {
                key: "category".to_string(),
                value: json!("tech"),
            },
            Filter::GreaterThan {
                key: "score".to_string(),
                value: 0.5,
            },
        ]);

        assert!(filter.matches(&metadata));
    }

    #[test]
    fn test_nested_keys() {
        let metadata = json!({
            "user": {
                "name": "Alice",
                "age": 30
            }
        });

        let filter = Filter::Equals {
            key: "user.name".to_string(),
            value: json!("Alice"),
        };

        assert!(filter.matches(&metadata));

        let filter2 = Filter::GreaterThan {
            key: "user.age".to_string(),
            value: 25.0,
        };

        assert!(filter2.matches(&metadata));
    }
}
