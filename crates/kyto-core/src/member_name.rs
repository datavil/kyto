//! Name rules of the store format, shared by the store and the plan.
//!
//! Does: say which rule of format.md a name breaks: §2.1 for layers, embeddings and
//! graphs (Zarr node names), §5.6 for table columns. kyto-store turns a broken rule
//! into `KY-E102` when it reads or writes; kyto-plan into `KY-E215` when a step names
//! a new member or column (task 4.2), so the sink never meets a name it cannot write.
//! Does not: check uniqueness (case-insensitive for members), which needs the group
//! or table the name goes into.

/// Longest member name in bytes (format.md §2.1).
pub const MAX_MEMBER_NAME_BYTES: usize = 64;

/// Windows device names, reserved with or without an extension, in any case.
const WINDOWS_DEVICES: [&str; 22] = [
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// The rule of format.md §2.1 the member name `name` breaks, or `None` if it is valid.
pub fn member_name_problem(name: &str) -> Option<String> {
    if name.is_empty() {
        return Some("the name is empty".to_owned());
    }
    if name.len() > MAX_MEMBER_NAME_BYTES {
        return Some(format!(
            "it is {} bytes long, more than {MAX_MEMBER_NAME_BYTES}",
            name.len()
        ));
    }
    if let Some(character) = name
        .chars()
        .find(|&character| !(character.is_ascii_alphanumeric() || "_.-".contains(character)))
    {
        return Some(format!("character {character:?} is not allowed"));
    }
    if name.starts_with('.') || name.starts_with("__") {
        return Some("it starts with '.' or '__'".to_owned());
    }
    if name.ends_with('.') {
        return Some("it ends with '.'".to_owned());
    }
    let stem = name.split('.').next().unwrap_or(name);
    if WINDOWS_DEVICES
        .iter()
        .any(|device| device.eq_ignore_ascii_case(stem))
    {
        return Some(format!("'{stem}' is a reserved Windows device name"));
    }
    None
}

/// The rule of format.md §5.6 the column name `name` breaks, or `None` if it is
/// valid: non-empty and without `:`, which feature names use as a separator.
pub fn column_name_problem(name: &str) -> Option<String> {
    if name.is_empty() {
        return Some("a column name is never empty".to_owned());
    }
    if name.contains(':') {
        return Some("':' is not allowed: feature names use it as a separator".to_owned());
    }
    None
}
