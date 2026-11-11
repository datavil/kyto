//! Build-time checks of the names a step writes (task 4.2, R6).
//!
//! Does: check a new layer, embedding or graph name against format.md §2.1 (syntax,
//! and no clash with another member of its group that differs only in case: macOS
//! and Windows file systems would merge them), and a new column name against §5.6
//! and the reserved ID and symbol columns, each as `KY-E215` naming the method and
//! argument, so a plan never asks its sink for a name the store cannot hold.
//! Does not: check names a step reads (the resolver, `KY-E109`/`KY-E210`), or
//! repeated names within one call (the step, which knows its arguments).

use kyto_core::{Axis, column_name_problem, member_name_problem};
use kyto_error::KytoError;

use crate::plan::Plan;
use crate::select_genes::quoted;

/// What format.md §2.1 accepts, for messages.
const MEMBER_NAME_RULES: &str = "1 to 64 characters from A-Z a-z 0-9 _ . - that do not start \
     with . or __, do not end with . and are not a Windows device name such as CON";

/// The group of the store a member name goes into.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MemberGroup {
    Layers,
    Embeddings,
    Graphs,
}

impl MemberGroup {
    fn kind(self) -> &'static str {
        match self {
            Self::Layers => "layer",
            Self::Embeddings => "embedding",
            Self::Graphs => "graph",
        }
    }
}

/// Reserved column names no plan writes, per table: both ID column names in either
/// table (format.md §5.2; the store writer refuses them too) and, in the gene table,
/// the symbols the scan read for `by='symbol'` (task 3.9).
pub(crate) fn reserved_columns(axis: Axis) -> &'static [&'static str] {
    match axis {
        Axis::Cells => &["cell_id", "gene_id"],
        Axis::Genes => &["cell_id", "gene_id", "symbol"],
    }
}

impl Plan {
    /// `KY-E215` unless `name`, the member of `group` that argument `parameter` of
    /// `method` (given as `given`) makes, may join this plan's members: valid under
    /// format.md §2.1, and equal to any member of its group that it matches ignoring
    /// case (equal names replace the member).
    pub(crate) fn check_member_name(
        &self,
        method: &str,
        parameter: &str,
        given: &str,
        group: MemberGroup,
        name: &str,
    ) -> Result<(), KytoError> {
        let kind = group.kind();
        let refuse = |problem: String, accepted: String| {
            Err(Self::invalid_argument(
                method,
                parameter,
                &quoted(given),
                &problem,
                &accepted,
            ))
        };
        if let Some(rule) = member_name_problem(name) {
            let problem = if name == given {
                format!("{rule} (format.md section 2.1)")
            } else {
                format!("the {kind} name '{name}' it makes breaks a rule: {rule}")
            };
            return refuse(problem, MEMBER_NAME_RULES.to_owned());
        }
        let schema = self.schema();
        let existing: Vec<&str> = match group {
            MemberGroup::Layers => schema.layers.iter().map(|m| m.name.as_str()).collect(),
            MemberGroup::Embeddings => schema.embeddings.iter().map(|m| m.name.as_str()).collect(),
            MemberGroup::Graphs => schema.graphs.iter().map(|m| m.name.as_str()).collect(),
        };
        if let Some(other) = existing
            .into_iter()
            .find(|other| *other != name && other.eq_ignore_ascii_case(name))
        {
            return refuse(
                format!(
                    "the {kind} '{name}' differs from the plan's {kind} '{other}' only in \
                     case, and {kind} names are compared ignoring case"
                ),
                format!(
                    "'{other}' (to replace that {kind}) or a name that differs in more than case"
                ),
            );
        }
        Ok(())
    }

    /// `KY-E215` unless `name` may be a column of the `axis` table that argument
    /// `parameter` of `method` writes: valid under format.md §5.6 and not reserved.
    pub(crate) fn check_column_name(
        method: &str,
        parameter: &str,
        axis: Axis,
        name: &str,
    ) -> Result<(), KytoError> {
        let reserved = reserved_columns(axis);
        let problem = column_name_problem(name).or_else(|| {
            reserved
                .contains(&name)
                .then(|| format!("'{name}' is reserved for the store"))
        });
        match problem {
            None => Ok(()),
            Some(problem) => Err(Self::invalid_argument(
                method,
                parameter,
                &quoted(name),
                &problem,
                &format!(
                    "a non-empty name without ':', other than {}",
                    reserved
                        .iter()
                        .map(|name| format!("'{name}'"))
                        .collect::<Vec<_>>()
                        .join(" and ")
                ),
            )),
        }
    }
}
