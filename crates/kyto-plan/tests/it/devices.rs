//! Task 9.1 Done when: `explain()` names the device per node on a machine with and
//! without a GPU.
//!
//! The devices found are an argument of the physical plan (`PhysicalOptions::devices`),
//! so one test plans for every machine: no GPU, an Apple GPU (Metal), an NVIDIA GPU
//! (CUDA), and both kinds. Each optimized node's line names its device; host steps stay
//! on the CPU; `auto` falls back to the CPU out loud (no GPU kernel exists before 9.2);
//! an explicit GPU is refused before the run with `KY-E504` (not found) or `KY-E501`
//! (no kernel). The `on` step round-trips through the plan JSON.
#![allow(clippy::unwrap_used)]

use crate::support;

use std::sync::Arc;

use kyto_core::{BinaryOp, CellExpr, Expr, Literal};
use kyto_plan::{
    Device, DeviceRequest, Devices, Explain, PhysicalOptions, PhysicalPlan, Plan, WgpuBackend,
};

use support::scan;

const NO_GPU: Devices = Devices::CPU_ONLY;
const MAC: Devices = Devices::new(false, Some(WgpuBackend::Metal));
const CUDA: Devices = Devices::new(true, None);
const BOTH: Devices = Devices::new(true, Some(WgpuBackend::Vulkan));

/// A metadata filter (a host step), a fused `normalize_total` + `log1p` pass and `pca`
/// (compute steps), then `on(device=request)`.
fn pipeline(request: DeviceRequest) -> Arc<Plan> {
    let score = Expr::binary(
        CellExpr::column("score"),
        BinaryOp::Gt,
        Literal::Float(-2.0),
    );
    scan()
        .filter_cells(&score.unwrap())
        .unwrap()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap()
        .pca(Some("logn"), "X_new", 2, 0, None)
        .unwrap()
        .on_device(request)
        .unwrap()
}

fn physical(plan: &Arc<Plan>, devices: Devices) -> PhysicalPlan {
    plan.physical(PhysicalOptions {
        devices,
        ..PhysicalOptions::default()
    })
}

/// The `[device=...` part of each optimized node line, by node name (top down).
fn devices_per_node(plan: &Arc<Plan>, devices: Devices) -> Vec<(String, String)> {
    let text = Explain::new(plan, &physical(plan, devices)).to_string();
    let optimized = text.split("optimized plan").nth(1).unwrap();
    optimized
        .lines()
        .filter(|line| line.trim_start().starts_with('#'))
        .map(|line| {
            let name = line.split_whitespace().nth(1).unwrap().to_owned();
            let device = line.split("[device=").nth(1).unwrap();
            let device = device.split(" passes=").next().unwrap().to_owned();
            (name, device)
        })
        .collect()
}

fn device_of<'a>(nodes: &'a [(String, String)], name: &str) -> &'a str {
    &nodes
        .iter()
        .find(|(node, _)| node == name)
        .unwrap_or_else(|| panic!("no {name} in {nodes:?}"))
        .1
}

#[test]
fn every_node_names_its_device_with_and_without_a_gpu() {
    for (devices, auto) in [
        (NO_GPU, "cpu (auto: no gpu found)"),
        (MAC, "cpu (auto: no gpu kernel)"),
        (CUDA, "cpu (auto: no gpu kernel)"),
        (BOTH, "cpu (auto: no gpu kernel)"),
    ] {
        let nodes = devices_per_node(&pipeline(DeviceRequest::Auto), devices);
        assert_eq!(nodes.len(), 4, "{nodes:?}");
        // Host steps run on the CPU, without a note.
        for host in ["FILTER_CELLS", "SCAN"] {
            assert_eq!(device_of(&nodes, host), "cpu", "{devices}: {host}");
        }
        // Compute steps: auto has no GPU kernel to use yet, and says so.
        for compute in ["FUSED_ELEMENTWISE", "PCA"] {
            assert_eq!(device_of(&nodes, compute), auto, "{devices}: {compute}");
        }
        // The header names the request and the devices found.
        let text = Explain::new(
            &pipeline(DeviceRequest::Auto),
            &physical(&pipeline(DeviceRequest::Auto), devices),
        )
        .to_string();
        assert!(
            text.contains(&format!("\n  device auto (found: {devices})\n")),
            "{text}"
        );
    }
}

#[test]
fn explicit_gpus_are_marked_and_refused_before_the_run() {
    let cases = [
        (
            DeviceRequest::Metal,
            MAC,
            "metal (no kernel: KY-E501 when run)",
            "KY-E501",
        ),
        (
            DeviceRequest::Wgpu,
            MAC,
            "metal (no kernel: KY-E501 when run)",
            "KY-E501",
        ),
        (
            DeviceRequest::Cuda,
            CUDA,
            "cuda (no kernel: KY-E501 when run)",
            "KY-E501",
        ),
        (
            DeviceRequest::Wgpu,
            BOTH,
            "vulkan (no kernel: KY-E501 when run)",
            "KY-E501",
        ),
        (
            DeviceRequest::Cuda,
            MAC,
            "cuda (not found: KY-E504 when run)",
            "KY-E504",
        ),
        (
            DeviceRequest::Metal,
            BOTH,
            "metal (not found: KY-E504 when run)",
            "KY-E504",
        ),
        (
            DeviceRequest::Wgpu,
            NO_GPU,
            "wgpu (not found: KY-E504 when run)",
            "KY-E504",
        ),
    ];
    for (request, devices, shown, code) in cases {
        let plan = pipeline(request);
        let nodes = devices_per_node(&plan, devices);
        assert_eq!(device_of(&nodes, "PCA"), shown, "{request} on {devices}");
        assert_eq!(device_of(&nodes, "FUSED_ELEMENTWISE"), shown);
        assert_eq!(device_of(&nodes, "SCAN"), "cpu");
        let error = physical(&plan, devices).device_error().unwrap();
        assert_eq!(error.code(), code, "{request} on {devices}");
        let message = error.to_string();
        if code == "KY-E501" {
            // The first step (from the scan up) without a kernel: the fused pass's first map.
            assert!(message.contains("'normalize_total'"), "{message}");
        } else {
            assert!(
                message.contains(&format!("(found: {devices})")),
                "{message}"
            );
            assert!(
                message.contains(&format!("device '{request}'")),
                "{message}"
            );
        }
    }
}

#[test]
fn cpu_plans_run_and_print_as_before() {
    for devices in [NO_GPU, MAC, BOTH] {
        for plan in [
            pipeline(DeviceRequest::Cpu),
            pipeline(DeviceRequest::Cpu).input().cloned().unwrap(),
        ] {
            let physical = physical(&plan, devices);
            assert!(physical.device_error().is_none());
            assert!(
                physical
                    .executions()
                    .iter()
                    .all(|execution| execution.device() == Some(Device::Cpu))
            );
            let text = Explain::new(&plan, &physical).to_string();
            // No device line for a CPU plan, and no notes.
            assert!(!text.contains("\n  device "), "{text}");
            assert!(!text.contains("(auto"), "{text}");
        }
        // auto falls back but still runs: no error.
        assert!(
            physical(&pipeline(DeviceRequest::Auto), devices)
                .device_error()
                .is_none()
        );
    }
    assert_eq!(scan().device_request(), DeviceRequest::Cpu);
}

#[test]
fn the_last_on_decides_for_the_whole_plan() {
    let plan = scan()
        .on_device(DeviceRequest::Cuda)
        .unwrap()
        .log1p(Some("counts"), "logn")
        .unwrap()
        .on_device(DeviceRequest::Auto)
        .unwrap();
    assert_eq!(plan.device_request(), DeviceRequest::Auto);
    // A step added after `on` keeps the plan's request.
    let later = plan.log1p(Some("logn"), "again").unwrap();
    assert_eq!(later.device_request(), DeviceRequest::Auto);
    let physical = physical(&later, MAC);
    assert_eq!(physical.device_request(), DeviceRequest::Auto);
    assert!(physical.device_error().is_none());
    assert_eq!(physical.estimate().device(), DeviceRequest::Auto);
    // `on` after a sink is refused like any step.
    let sunk = scan().sink(kyto_plan::Sink::Zarr {
        path: "out.kyto".to_owned(),
    });
    let refused = sunk.unwrap().on_device(DeviceRequest::Auto).err().unwrap();
    assert_eq!(refused.code(), "KY-E216");
}

#[test]
fn the_on_step_round_trips_through_the_plan_json() {
    for request in [
        DeviceRequest::Cpu,
        DeviceRequest::Auto,
        DeviceRequest::Cuda,
        DeviceRequest::Wgpu,
        DeviceRequest::Metal,
    ] {
        let plan = pipeline(request);
        let json = plan.to_json();
        assert!(
            json.contains(r#""method": "on""#) || json.contains(r#""method":"on""#),
            "{json}"
        );
        let read = Plan::from_json(&json).unwrap();
        assert_eq!(read.to_json(), json);
        assert_eq!(read.device_request(), request);
        assert!(kyto_plan::validate_plan_json(&json).is_empty());
    }
    let json = pipeline(DeviceRequest::Auto)
        .to_json()
        .replace(r#""device": "auto""#, r#""device": "gpu""#)
        .replace(r#""device":"auto""#, r#""device":"gpu""#);
    let error = Plan::from_json(&json).err().unwrap();
    assert_eq!(error.code(), "KY-E215");
    assert!(error.to_string().contains("'gpu'"), "{error}");
}

#[test]
fn an_on_between_maps_does_not_stop_their_fusion() {
    let fused = |plan: Arc<Plan>| {
        let physical = physical(&plan, MAC);
        let names: Vec<&str> = physical
            .optimized()
            .nodes()
            .iter()
            .map(|node| node.name())
            .collect();
        (names, physical.estimate().passes())
    };
    let middle = scan()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .on_device(DeviceRequest::Auto)
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap();
    let end = scan()
        .normalize_total(Some(1e4), Some("counts"), "norm")
        .unwrap()
        .log1p(Some("norm"), "logn")
        .unwrap()
        .on_device(DeviceRequest::Auto)
        .unwrap();
    assert_eq!(fused(middle.clone()), fused(end));
    assert_eq!(fused(middle).0, ["SCAN", "FUSED_ELEMENTWISE"]);
}
