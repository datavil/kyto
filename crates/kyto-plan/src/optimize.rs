//! The optimizer (task 4.4): from the logical plans of one or more terminals to one
//! [`OptimizedPlan`].
//!
//! Does: take what each terminal reads ([`Output`]), lower the logical plans to the
//! optimized arena (one chain per output, or with common sub-plan reuse, one node per
//! distinct sub-plan: `optimize_reuse.rs`) and run the rules in a fixed order:
//! projection pushdown (`optimize_projection.rs`: leave out nodes whose outputs nothing
//! reads, and read only what is needed from the store), elementwise fusion
//! (`optimize_fusion.rs`), predicate pushdown (`optimize_predicate.rs`: cell filters
//! move below per-cell nodes, stopping at cross-cell ones), fusion again (maps a moved
//! filter separated), then projection pushdown again (fusion and pushdown change what
//! each node must keep). Fusion first keeps `normalize_total` to the median and the
//! `log1p` after it in one pass, with a filter above them, rather than splitting them
//! around the filter. Each rule can be
//! turned off ([`OptimizeOptions`], like Polars' optimization flags) to compare plans
//! and to test the rules one at a time.
//! Does not: change the logical plan (immutable, R2; the plan JSON writes it, not the
//! optimized one), choose devices or passes (4.5), or run anything (5.x).
//!
//! Every rule keeps results equal to the logical plan's (bit for bit: no rule reorders
//! floating-point arithmetic). One exception, recorded in deviations.md: a step whose
//! output no terminal reads is not run, so an error it would raise only when run (on
//! data) is not raised.

use std::sync::Arc;

use kyto_error::KytoError;

use crate::members::{Member, Members, all_members, has_member};
use crate::optimized_plan::OptimizedPlan;
use crate::plan::Plan;
use crate::{optimize_fusion, optimize_predicate, optimize_projection, optimize_reuse};

/// Which optimizer rules run. All do by default.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OptimizeOptions {
    /// Compute a sub-plan shared by several outputs once.
    pub reuse_sub_plans: bool,
    /// Leave out nodes whose outputs nothing reads; read only the members needed.
    pub projection_pushdown: bool,
    /// Run consecutive value-by-value maps (`normalize_total`, `log1p`) in one pass.
    pub fuse_elementwise: bool,
    /// Move cell filters below per-cell nodes, stopping at cross-cell ones.
    pub predicate_pushdown: bool,
}

impl OptimizeOptions {
    /// Every rule on.
    pub const ALL: Self = Self {
        reuse_sub_plans: true,
        projection_pushdown: true,
        fuse_elementwise: true,
        predicate_pushdown: true,
    };

    /// Every rule off: the logical plans as they are, one chain per output.
    pub const NONE: Self = Self {
        reuse_sub_plans: false,
        projection_pushdown: false,
        fuse_elementwise: false,
        predicate_pushdown: false,
    };
}

impl Default for OptimizeOptions {
    fn default() -> Self {
        Self::ALL
    }
}

/// What a terminal reads: a plan and its members.
#[derive(Debug, Clone)]
pub struct Output {
    pub(crate) plan: Arc<Plan>,
    pub(crate) reads: Members,
}

impl Output {
    /// Every member of `plan`'s output (a sink, `collect`).
    pub fn all(plan: &Arc<Plan>) -> Self {
        Self {
            plan: Arc::clone(plan),
            reads: all_members(plan.schema()),
        }
    }

    /// Only `members` of `plan`'s output (a handle's terminal: one layer, one table's
    /// columns). The ID columns always come along.
    ///
    /// # Errors
    /// `KY-E109` for a member the plan does not output.
    pub fn members(
        plan: &Arc<Plan>,
        members: impl IntoIterator<Item = Member>,
    ) -> Result<Self, KytoError> {
        let reads: Members = members.into_iter().collect();
        if let Some(missing) = reads
            .iter()
            .find(|member| !has_member(plan.schema(), member))
        {
            let (kind, name) = match missing {
                Member::Layer(name) => ("layer", name),
                Member::CellColumn(name) => ("cell column", name),
                Member::GeneColumn(name) => ("gene column", name),
                Member::Embedding(name) => ("embedding", name),
                Member::Graph(name) => ("graph", name),
            };
            return Err(plan.missing_member(kind, name));
        }
        Ok(Self {
            plan: Arc::clone(plan),
            reads,
        })
    }

    /// The plan read.
    pub fn plan(&self) -> &Arc<Plan> {
        &self.plan
    }

    /// The members read.
    pub fn reads(&self) -> &Members {
        &self.reads
    }
}

/// The optimized plan of `outputs`, with the rules `options` turns on.
pub fn optimize(outputs: &[Output], options: OptimizeOptions) -> OptimizedPlan {
    let mut plan = optimize_reuse::lower(outputs, options.reuse_sub_plans);
    if options.projection_pushdown {
        plan = optimize_projection::push_down(plan);
    }
    if options.fuse_elementwise {
        plan = optimize_fusion::fuse(plan);
    }
    if options.predicate_pushdown {
        plan = optimize_predicate::push_down(plan);
        if options.fuse_elementwise {
            // Filters moved out from between maps let those maps fuse.
            plan = optimize_fusion::fuse(plan);
        }
    }
    if options.projection_pushdown {
        plan = optimize_projection::push_down(plan);
    }
    plan
}

impl Plan {
    /// This plan optimized with every rule, for a terminal that reads all of its
    /// output. The plan itself is unchanged.
    pub fn optimize(self: &Arc<Self>) -> OptimizedPlan {
        optimize(&[Output::all(self)], OptimizeOptions::ALL)
    }
}
