//! Close-match suggestions for unknown names (task 8.2): the `suggestions` field of
//! `KY-E109` and `KY-E210` (`; did you mean 'CD3E', 'CD3D', 'CD3G'?`).
//!
//! Does: score each candidate against the unknown name with Jaro-Winkler and keep the
//! [`MAX_SUGGESTIONS`] best scoring at least [`MIN_SIMILARITY`], best first, ties in
//! candidate order (table or schema order, so the result is deterministic, R11).
//! Names are compared case-insensitively (a case typo is the most common one, and
//! matching itself is exact) and with every run of ASCII digits as one symbol, so a
//! number is never half-matched: `CD3` scores 0.94 against `CD3E` but 0.82 against
//! `CD34` (a different number, still above the cut-off, so it fills a free place), and
//! `PC1` is as far from `PC10` as from `PC2`. Plain character Jaro-Winkler ties `CD3E`
//! with `CD34`, `CD38`, `CD36`, ... on pbmc3k. Winkler's prefix boost applies only
//! from a Jaro similarity of 0.7, as Winkler (1990) defines it.
//! Does not: decide which names are candidates (the caller, by kind), or run on any
//! path but an error's.

/// At most this many suggestions per error.
pub const MAX_SUGGESTIONS: usize = 3;

/// The least Jaro-Winkler similarity (0 to 1) a candidate needs to be suggested.
pub const MIN_SIMILARITY: f64 = 0.8;

/// Winkler's prefix weight and the longest prefix it rewards (the usual 0.1 and 4).
const PREFIX_WEIGHT: f64 = 0.1;
const MAX_PREFIX: usize = 4;
/// The Jaro similarity from which the prefix boost applies (Winkler's 0.7).
const BOOST_THRESHOLD: f64 = 0.7;

/// The names among `candidates` closest to `name`, best first (see the module doc).
pub fn suggest<'a>(name: &str, candidates: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    closest(
        name,
        candidates
            .into_iter()
            .map(|candidate| (candidate, candidate.to_owned())),
    )
}

/// The labels of the candidates whose keys are closest to `name`, best first. Each
/// candidate is `(key, label)`: the key is scored, the label is returned (such as
/// `CD3E (gene symbol)` for the key `CD3E`). A label already kept is not repeated.
pub fn closest<'a, L: PartialEq>(
    name: &str,
    candidates: impl IntoIterator<Item = (&'a str, L)>,
) -> Vec<L> {
    let target = symbols(name);
    let mut scored: Vec<(f64, L)> = candidates
        .into_iter()
        .filter_map(|(key, label)| {
            let score = jaro_winkler(&target, &symbols(key));
            (score >= MIN_SIMILARITY).then_some((score, label))
        })
        .collect();
    // Stable: equal scores keep candidate order. Scores are never NaN.
    scored.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut kept: Vec<L> = Vec::with_capacity(MAX_SUGGESTIONS);
    for (_, label) in scored {
        if kept.len() == MAX_SUGGESTIONS {
            break;
        }
        if !kept.contains(&label) {
            kept.push(label);
        }
    }
    kept
}

/// Jaro-Winkler similarity of two names, from 0 (nothing alike) to 1 (equal up to
/// case), on the symbols of the module doc.
pub fn similarity(left: &str, right: &str) -> f64 {
    jaro_winkler(&symbols(left), &symbols(right))
}

/// `name` lowercased and cut into symbols: one per character, one per run of ASCII
/// digits.
fn symbols(name: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut previous_digit = false;
    for character in name.chars().flat_map(char::to_lowercase) {
        let digit = character.is_ascii_digit();
        match out.last_mut() {
            Some(last) if digit && previous_digit => last.push(character),
            _ => out.push(character.to_string()),
        }
        previous_digit = digit;
    }
    out
}

fn jaro_winkler(left: &[String], right: &[String]) -> f64 {
    let jaro = jaro(left, right);
    if jaro < BOOST_THRESHOLD {
        return jaro;
    }
    let prefix = left
        .iter()
        .zip(right)
        .take(MAX_PREFIX)
        .take_while(|(a, b)| a == b)
        .count();
    jaro + prefix as f64 * PREFIX_WEIGHT * (1.0 - jaro)
}

fn jaro(left: &[String], right: &[String]) -> f64 {
    if left.is_empty() && right.is_empty() {
        return 1.0;
    }
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    let window = (left.len().max(right.len()) / 2).saturating_sub(1);
    let mut right_matched = vec![false; right.len()];
    let mut left_matches: Vec<&String> = Vec::new();
    for (index, symbol) in left.iter().enumerate() {
        let start = index.saturating_sub(window);
        let end = (index + window + 1).min(right.len());
        for other in start..end {
            if !right_matched[other] && right[other] == *symbol {
                right_matched[other] = true;
                left_matches.push(symbol);
                break;
            }
        }
    }
    let matches = left_matches.len();
    if matches == 0 {
        return 0.0;
    }
    let right_matches = right
        .iter()
        .zip(&right_matched)
        .filter_map(|(symbol, matched)| matched.then_some(symbol));
    let half_transpositions = left_matches
        .iter()
        .zip(right_matches)
        .filter(|(a, b)| **a != *b)
        .count();
    let matches = matches as f64;
    let transpositions = (half_transpositions / 2) as f64;
    (matches / left.len() as f64
        + matches / right.len() as f64
        + (matches - transpositions) / matches)
        / 3.0
}

#[cfg(test)]
mod tests {
    use super::{closest, similarity, suggest, symbols};

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-9, "{actual} != {expected}");
    }

    #[test]
    fn classic_jaro_winkler_values() {
        // Winkler (1990) and the usual textbook pairs.
        close(similarity("MARTHA", "MARHTA"), 0.961_111_111_111_111_1);
        close(similarity("DWAYNE", "DUANE"), 0.84);
        close(similarity("DIXON", "DICKSONX"), 0.813_333_333_333_333_3);
        close(similarity("", ""), 1.0);
        close(similarity("a", ""), 0.0);
        close(similarity("abc", "xyz"), 0.0);
    }

    #[test]
    fn case_is_ignored_and_digit_runs_are_one_symbol() {
        close(similarity("cd3e", "CD3E"), 1.0);
        assert_eq!(symbols("CD34x12"), ["c", "d", "34", "x", "12"]);
        close(similarity("CD3", "CD3E"), 0.941_666_666_666_666_7);
        close(similarity("CD3", "CD34"), 0.822_222_222_222_222_2);
        // Below a Jaro of 0.7 the shared prefix adds nothing: 'abcdwxyz' vs 'abcdpqrs'.
        close(similarity("abcdwxyz", "abcdpqrs"), 2.0 / 3.0);
        close(similarity("PC1", "PC10"), similarity("PC1", "PC2"));
    }

    #[test]
    fn top_three_best_first_ties_in_candidate_order() {
        let genes = [
            "CD34", "CD302", "CD38", "CD36", "CD3E", "CD3D", "CD3G", "CD3EAP", "CD37",
        ];
        assert_eq!(suggest("CD3", genes), ["CD3E", "CD3D", "CD3G"]);
        assert_eq!(suggest("count", ["log1p", "counts"]), ["counts"]);
        assert!(suggest("x", ["counts", "log1p"]).is_empty());
        assert!(suggest("CD3", []).is_empty());
    }

    #[test]
    fn labels_are_not_repeated() {
        let found = closest("CD3", [("CD3E", "a"), ("CD3E", "a"), ("CD3D", "b")]);
        assert_eq!(found, ["a", "b"]);
    }
}
