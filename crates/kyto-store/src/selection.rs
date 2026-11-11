//! Cell and gene selections of the tile reader, split by block (format.md §4.7, §4.8).
//!
//! Does: hold a selection of table positions on one axis ([`AxisSelection`]), check it
//! (in range, sorted, unique) and map it to the blocks it touches, with the selected
//! local positions of each block and their positions in the selection.
//! Does not: resolve names, evaluate predicates or canonicalise a user's order or
//! repeats; the plan does that (format.md §4.8). Reads nothing.

use std::ops::Range;

use kyto_error::KytoError;

use crate::table_axis::TableAxis;

/// The cells or genes the tile reader reads, as table positions (format.md §4.8).
///
/// Positions are rows of `cells.parquet` or `genes.parquet`, which are the matrix rows
/// and columns (format.md §4.1). A selection is canonical: [`AxisSelection::Indices`]
/// must be strictly increasing, so it has no repeats. Output follows table order; a
/// user selection in another order or with repeats is canonicalised by the plan, and
/// any reordering for output happens at the terminal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AxisSelection {
    /// Every position of the axis.
    All,
    /// Positions `start..end`, with `start <= end <= n` (empty when equal).
    Range(Range<u64>),
    /// Strictly increasing positions, each below the axis length (empty is allowed).
    Indices(Vec<u64>),
}

/// The selected local positions of one block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Picks {
    /// Local positions `start..end`.
    Range {
        /// First selected local position.
        start: u32,
        /// One past the last selected local position.
        end: u32,
    },
    /// Strictly increasing local positions.
    List(Vec<u32>),
}

/// The selected positions inside one block of an axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct BlockPick {
    /// The block number (`floor(position / block size)`).
    pub block: u64,
    /// Position in the selection of this block's first selected position.
    pub first_position: u64,
    /// The selected local positions.
    pub picks: Picks,
}

impl BlockPick {
    /// Number of selected positions in the block.
    pub(crate) fn len(&self) -> u64 {
        match &self.picks {
            Picks::Range { start, end } => u64::from(end - start),
            Picks::List(locals) => locals.len() as u64,
        }
    }

    /// Position in the selection of local position `local`, if it is selected.
    pub(crate) fn position(&self, local: u32) -> Option<u64> {
        let offset = match &self.picks {
            Picks::Range { start, end } => (*start..*end)
                .contains(&local)
                .then(|| u64::from(local - start)),
            Picks::List(locals) => locals.binary_search(&local).ok().map(|at| at as u64),
        }?;
        Some(self.first_position + offset)
    }

    /// The selected local positions, increasing.
    pub(crate) fn locals(&self) -> Vec<u32> {
        match &self.picks {
            Picks::Range { start, end } => (*start..*end).collect(),
            Picks::List(locals) => locals.clone(),
        }
    }

    /// The local rows `first..last + 1` a read of the selected positions spans
    /// (`0..0` when none is selected).
    pub(crate) fn span(&self) -> (u32, u32) {
        match &self.picks {
            Picks::Range { start, end } => (*start, *end),
            Picks::List(locals) => match (locals.first(), locals.last()) {
                (Some(&first), Some(&last)) => (first, last + 1),
                _ => (0, 0),
            },
        }
    }

    /// Split into picks of at most `max_len` selected positions each, in order, on
    /// the same block (task 5.5: blocks smaller than the store's). `max_len` is at
    /// least 1.
    pub(crate) fn split(&self, max_len: u64) -> Vec<BlockPick> {
        let max_len = max_len.max(1);
        if self.len() <= max_len {
            return vec![self.clone()];
        }
        // max_len < len <= block size <= 2^32 - 1.
        let step = max_len as u32;
        let part = |offset: u64, picks: Picks| BlockPick {
            block: self.block,
            first_position: self.first_position + offset,
            picks,
        };
        match &self.picks {
            Picks::Range { start, end } => (*start..*end)
                .step_by(step as usize)
                .map(|from| {
                    let to = from.saturating_add(step).min(*end);
                    part(
                        u64::from(from - start),
                        Picks::Range {
                            start: from,
                            end: to,
                        },
                    )
                })
                .collect(),
            Picks::List(locals) => locals
                .chunks(step as usize)
                .enumerate()
                .map(|(index, chunk)| part(index as u64 * max_len, Picks::List(chunk.to_vec())))
                .collect(),
        }
    }
}

/// Check `selection` against an axis of length `n` and split it into blocks of
/// `block_size`, keeping only blocks with at least one selected position.
///
/// `block_size` is `C` or `G`, which [`crate::TileGrid`] keeps at most `2^32 - 1`, so
/// local positions fit `u32`.
///
/// # Errors
/// `KY-E114` if a position is out of range or the indices are not strictly increasing.
pub(crate) fn block_picks(
    selection: &AxisSelection,
    axis: TableAxis,
    n: u64,
    block_size: u64,
) -> Result<Vec<BlockPick>, KytoError> {
    let invalid = |problem: String| KytoError::InvalidSelection {
        axis: axis.as_str().to_owned(),
        problem,
        n_axis: n,
    };
    match selection {
        AxisSelection::All => Ok(range_picks(0, n, n, block_size)),
        AxisSelection::Range(range) => {
            if range.start > range.end {
                return Err(invalid(format!(
                    "range {}..{} starts after it ends",
                    range.start, range.end
                )));
            }
            if range.end > n {
                return Err(invalid(format!(
                    "range {}..{} ends after the axis length {n}",
                    range.start, range.end
                )));
            }
            Ok(range_picks(range.start, range.end, n, block_size))
        }
        AxisSelection::Indices(indices) => {
            let mut picks: Vec<BlockPick> = Vec::new();
            for (position, &index) in indices.iter().enumerate() {
                if index >= n {
                    return Err(invalid(format!(
                        "position {position} is {index}, not below the axis length {n}"
                    )));
                }
                if position > 0 && index <= indices[position - 1] {
                    let previous = indices[position - 1];
                    return Err(invalid(if index == previous {
                        format!("position {position} repeats {index}")
                    } else {
                        format!("position {position} is {index}, which does not follow {previous}")
                    }));
                }
                let block = index / block_size;
                // index % block_size < block_size <= 2^32 - 1.
                let local = (index % block_size) as u32;
                match picks.last_mut() {
                    Some(BlockPick {
                        block: last,
                        picks: Picks::List(locals),
                        ..
                    }) if *last == block => locals.push(local),
                    _ => picks.push(BlockPick {
                        block,
                        first_position: position as u64,
                        picks: Picks::List(vec![local]),
                    }),
                }
            }
            Ok(picks)
        }
    }
}

/// Blocks of the valid range `start..end` of an axis of length `n`.
fn range_picks(start: u64, end: u64, n: u64, block_size: u64) -> Vec<BlockPick> {
    if start >= end {
        return Vec::new();
    }
    let first_block = start / block_size;
    let last_block = (end - 1) / block_size;
    (first_block..=last_block)
        .map(|block| {
            let block_start = block * block_size;
            let block_end = (block_start + block_size).min(n);
            let from = start.max(block_start);
            let to = end.min(block_end);
            BlockPick {
                block,
                first_position: from - start,
                // Both are at most block_size <= 2^32 - 1.
                picks: Picks::Range {
                    start: (from - block_start) as u32,
                    end: (to - block_start) as u32,
                },
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{AxisSelection, BlockPick, Picks, block_picks};
    use crate::table_axis::TableAxis;

    fn range(block: u64, first_position: u64, start: u32, end: u32) -> BlockPick {
        BlockPick {
            block,
            first_position,
            picks: Picks::Range { start, end },
        }
    }

    #[test]
    fn all_and_ranges_split_by_block() {
        let all = block_picks(&AxisSelection::All, TableAxis::Genes, 7, 3).unwrap();
        assert_eq!(
            all,
            vec![range(0, 0, 0, 3), range(1, 3, 0, 3), range(2, 6, 0, 1)]
        );
        let part = block_picks(&AxisSelection::Range(2..5), TableAxis::Genes, 7, 3).unwrap();
        assert_eq!(part, vec![range(0, 0, 2, 3), range(1, 1, 0, 2)]);
        assert!(
            block_picks(&AxisSelection::Range(4..4), TableAxis::Genes, 7, 3)
                .unwrap()
                .is_empty()
        );
        assert!(
            block_picks(&AxisSelection::All, TableAxis::Cells, 0, 3)
                .unwrap()
                .is_empty()
        );
    }

    #[test]
    fn indices_group_by_block_with_positions() {
        let picks = block_picks(
            &AxisSelection::Indices(vec![1, 2, 7, 9]),
            TableAxis::Genes,
            10,
            3,
        )
        .unwrap();
        assert_eq!(picks.len(), 3);
        assert_eq!((picks[0].block, picks[0].first_position), (0, 0));
        assert_eq!(picks[0].picks, Picks::List(vec![1, 2]));
        assert_eq!((picks[1].block, picks[1].first_position), (2, 2));
        assert_eq!(picks[1].locals(), vec![1]);
        assert_eq!(picks[1].len(), 1);
        // Gene 9 sits alone in block 3.
        assert_eq!((picks[2].block, picks[2].first_position), (3, 3));
        assert_eq!(picks[2].locals(), vec![0]);
    }

    #[test]
    fn positions_map_back_to_the_selection() {
        let list = block_picks(
            &AxisSelection::Indices(vec![4, 5, 8]),
            TableAxis::Genes,
            9,
            3,
        )
        .unwrap();
        assert_eq!(list[0].position(1), Some(0));
        assert_eq!(list[0].position(2), Some(1));
        assert_eq!(list[0].position(0), None);
        assert_eq!(list[1].position(2), Some(2));
        let ranged = block_picks(&AxisSelection::Range(4..8), TableAxis::Genes, 9, 3).unwrap();
        assert_eq!(ranged[0].position(0), None);
        assert_eq!(ranged[0].position(1), Some(0));
        assert_eq!(ranged[1].position(1), Some(3));
        assert_eq!(ranged[1].position(2), None);
    }

    #[test]
    fn picks_split_into_parts_of_at_most_max_len() {
        let ranged = range(4, 9, 1, 8);
        assert_eq!(ranged.split(7), vec![ranged.clone()]);
        assert_eq!(ranged.split(100), vec![ranged.clone()]);
        assert_eq!(
            ranged.split(3),
            vec![range(4, 9, 1, 4), range(4, 12, 4, 7), range(4, 15, 7, 8)]
        );
        assert_eq!(ranged.span(), (1, 8));
        let listed = BlockPick {
            block: 2,
            first_position: 5,
            picks: Picks::List(vec![0, 3, 4, 9, 11]),
        };
        let parts = listed.split(2);
        assert_eq!(parts.len(), 3);
        assert_eq!(parts[0].picks, Picks::List(vec![0, 3]));
        assert_eq!((parts[1].first_position, parts[1].span()), (7, (4, 10)));
        assert_eq!((parts[2].first_position, parts[2].span()), (9, (11, 12)));
        assert!(parts.iter().all(|part| part.block == 2));
        // Zero is treated as 1.
        assert_eq!(range(0, 0, 0, 2).split(0).len(), 2);
    }

    fn problem(selection: AxisSelection, n: u64) -> String {
        let error = block_picks(&selection, TableAxis::Genes, n, 3).unwrap_err();
        assert_eq!(error.code(), "KY-E114");
        error.to_string()
    }

    #[test]
    fn bad_selections_are_rejected() {
        assert!(
            problem(AxisSelection::Indices(vec![1, 9]), 9)
                .contains("invalid genes selection: position 1 is 9, not below the axis length 9")
        );
        assert!(problem(AxisSelection::Indices(vec![1, 1]), 9).contains("position 1 repeats 1"));
        assert!(
            problem(AxisSelection::Indices(vec![5, 2]), 9)
                .contains("position 1 is 2, which does not follow 5")
        );
        assert!(problem(AxisSelection::Range(2..10), 9).contains("range 2..10 ends after"));
        #[allow(clippy::reversed_empty_ranges)]
        let reversed = AxisSelection::Range(5..2);
        assert!(problem(reversed, 9).contains("range 5..2 starts after it ends"));
        assert!(
            problem(AxisSelection::Indices(vec![0]), 0).contains("not below the axis length 0")
        );
    }
}
