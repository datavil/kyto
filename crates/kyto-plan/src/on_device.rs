//! The node that sets the device the plan runs on (`on(device=...)`, task 9.1).
//!
//! Does: hold the requested device ([`DeviceRequest`]). The last `on` of a chain sets
//! the device of the whole plan, the steps before it too (R7: the device is an
//! argument of the plan, never a global setting); a plan without one runs on the CPU.
//! Its output schema is its input's; it changes no axis and reads nothing. The
//! optimizer leaves it out of the optimized plan (its request is
//! `OptimizedPlan::device_request`), so it never keeps maps around it from fusing.
//! Does not: place each node on a device (`device_placement.rs`, from the request and
//! the devices found) or check that the device exists (that happens when the plan
//! runs, `KY-E504`, so a plan built on one machine can be explained on another).

use std::fmt;
use std::sync::Arc;

use kyto_core::Axis;
use kyto_error::KytoError;
use serde::{Deserialize, Serialize};

use crate::device_placement::DeviceRequest;
use crate::node::{CellScope, LogicalNode, Transform};
use crate::plan::Plan;
use crate::plan_schema::PlanSchema;

/// Sets the device of the whole plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OnDevice {
    device: DeviceRequest,
}

impl OnDevice {
    /// The device asked for.
    pub fn device(&self) -> DeviceRequest {
        self.device
    }
}

impl LogicalNode for OnDevice {
    fn name(&self) -> &'static str {
        "ON_DEVICE"
    }

    fn method(&self) -> &'static str {
        "on"
    }

    fn changes_axis(&self) -> Option<Axis> {
        None
    }

    fn cell_scope(&self) -> CellScope {
        CellScope::PerCell
    }

    fn describe(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{:?}", self.device.name())
    }
}

impl Transform for OnDevice {
    fn output_schema(&self, input: &PlanSchema) -> PlanSchema {
        input.clone()
    }

    fn sets_device(&self) -> Option<DeviceRequest> {
        Some(self.device)
    }
}

impl Plan {
    /// This plan to run on `device` (a new plan; `self` is unchanged). The last `on` of
    /// a chain decides the device of every step of the plan.
    ///
    /// # Errors
    /// `KY-E216` if this plan ends in a sink.
    pub fn on_device(self: &Arc<Self>, device: DeviceRequest) -> Result<Arc<Self>, KytoError> {
        self.check_open("on")?;
        Ok(self.extend(OnDevice { device }, Self::OnDevice))
    }

    /// The device the plan asks for: its last `on`, or the CPU without one.
    pub fn device_request(&self) -> DeviceRequest {
        self.state().device
    }
}
