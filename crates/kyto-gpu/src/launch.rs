//! Running the 9.2 kernels on a GPU: the host side of `kernels.rs`.
//!
//! Does: open a device's CubeCL client ([`DeviceKernels::open`]), copy a block's row
//! pointer and values to the device, launch one kernel, and copy its output back
//! (one round trip per call; keeping blocks on the device between steps is
//! milestone M6's). Each method gives what the CPU kernel of the same name in
//! kyto-exec gives, within 1e-6 relative (`kernels.rs` says where the bits differ).
//! Does not: choose devices (kyto-plan's `device_placement.rs`) or decide which steps
//! call it (kyto-exec's `elementwise.rs`).
//!
//! Built without a GPU feature, [`DeviceKernels::open`] always fails and no value of
//! the type exists, so the default build carries no GPU code.

use kyto_error::KytoError;

use crate::device::Device;

/// The values of a block's rows: a layer's `f32` values or `u32` counts.
#[derive(Debug, Clone, Copy)]
pub enum RowValues<'a> {
    /// `float32` values.
    F32(&'a [f32]),
    /// `uint32` counts.
    U32(&'a [u32]),
}

impl RowValues<'_> {
    /// The number of values.
    pub fn len(self) -> usize {
        match self {
            Self::F32(values) => values.len(),
            Self::U32(values) => values.len(),
        }
    }

    /// Whether there are none.
    pub fn is_empty(self) -> bool {
        self.len() == 0
    }
}

/// The 9.2 kernels on one GPU. Cheap to clone and shared by the executor's threads.
#[derive(Clone)]
pub struct DeviceKernels {
    device: Device,
    #[cfg(any(feature = "wgpu", feature = "cuda"))]
    client: cubecl::prelude::Client,
    /// No value exists in a build without a GPU runtime.
    #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
    never: std::convert::Infallible,
}

impl std::fmt::Debug for DeviceKernels {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceKernels")
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

/// A failure of `operation` on `device`, as `KY-E505`.
fn failure(device: Device, operation: &str, problem: impl ToString) -> KytoError {
    KytoError::DeviceFailure {
        device: device.name().to_owned(),
        operation: operation.to_owned(),
        problem: problem.to_string(),
    }
}

// Without a GPU runtime the methods' bodies are `match self.never {}`.
#[cfg_attr(not(any(feature = "wgpu", feature = "cuda")), allow(unused_variables))]
impl DeviceKernels {
    /// The kernels on `device` (a GPU the probe found), for `operation` (named in the
    /// error). Compiles nothing yet: each kernel is compiled on its first launch and
    /// cached by CubeCL.
    ///
    /// # Errors
    /// `KY-E505` if the device's runtime is not built in, `device` is the CPU, or the
    /// runtime cannot start on it.
    pub fn open(device: Device, operation: &str) -> Result<Self, KytoError> {
        #[cfg(any(feature = "wgpu", feature = "cuda"))]
        {
            let client = open_client(device, operation)?;
            Ok(Self { device, client })
        }
        #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
        Err(failure(
            device,
            operation,
            "this kyto was built without a GPU feature",
        ))
    }

    /// The device the kernels run on.
    pub fn device(&self) -> Device {
        self.device
    }

    /// Each row's sum, in `f64`: a `u32` row exactly (as the CPU's, below 2^53), an
    /// `f32` row as a compensated `f32` pair (`kernels::row_sums_f32`), within about
    /// 1e-11 relative of the exact sum (5e-12 measured on Metal).
    ///
    /// # Errors
    /// `KY-E505` if a copy or the kernel fails; `operation` names the plan step.
    pub fn row_sums(
        &self,
        indptr: &[u64],
        values: RowValues<'_>,
        operation: &str,
    ) -> Result<Vec<f64>, KytoError> {
        #[cfg(any(feature = "wgpu", feature = "cuda"))]
        return run::row_sums(self, indptr, values, operation);
        #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
        match self.never {}
    }

    /// Each row's number of nonzero values (a stored zero is not counted, a NaN is).
    ///
    /// # Errors
    /// `KY-E505` if a copy or the kernel fails.
    pub fn row_nnz(
        &self,
        indptr: &[u64],
        values: RowValues<'_>,
        operation: &str,
    ) -> Result<Vec<u64>, KytoError> {
        #[cfg(any(feature = "wgpu", feature = "cuda"))]
        return run::row_nnz(self, indptr, values, operation);
        #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
        match self.never {}
    }

    /// Each value divided by its row's divisor in `f32`, written to `out`.
    ///
    /// # Errors
    /// `KY-E505` if a copy or the kernel fails.
    ///
    /// # Panics
    /// If `divisors` has not one divisor per row or `out` is not as long as `values`
    /// (as the CPU kernel).
    pub fn divide_rows(
        &self,
        indptr: &[u64],
        values: RowValues<'_>,
        divisors: &[f32],
        out: &mut [f32],
        operation: &str,
    ) -> Result<(), KytoError> {
        #[cfg(any(feature = "wgpu", feature = "cuda"))]
        return run::divide_rows(self, indptr, values, divisors, out, operation);
        #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
        match self.never {}
    }

    /// `log1p` of each value (a `u32` rounded to `f32` first), written to `out`.
    ///
    /// # Errors
    /// `KY-E505` if a copy or the kernel fails.
    ///
    /// # Panics
    /// If `out` is not as long as `values`.
    pub fn log1p(
        &self,
        values: RowValues<'_>,
        out: &mut [f32],
        operation: &str,
    ) -> Result<(), KytoError> {
        #[cfg(any(feature = "wgpu", feature = "cuda"))]
        return run::log1p(self, values, out, operation);
        #[cfg(not(any(feature = "wgpu", feature = "cuda")))]
        match self.never {}
    }
}

/// The CubeCL client of `device`, on the graphics API the probe found (Metal, Vulkan
/// or DX12 for wgpu).
///
/// # Errors
/// `KY-E505` (naming `operation`) if its runtime is not built in, the machine does not
/// have it, or it fails to start (CubeCL panics then; the panic is caught here).
#[cfg(any(feature = "wgpu", feature = "cuda"))]
fn open_client(device: Device, operation: &str) -> Result<cubecl::prelude::Client, KytoError> {
    use crate::device::WgpuBackend;
    use cubecl::device::WgpuDeviceKind;
    let kind = WgpuDeviceKind::default;
    let named = match device {
        Device::Cpu => return Err(failure(device, operation, "the CPU has no GPU kernels")),
        Device::Cuda => cubecl::Device::cuda(0),
        Device::Wgpu(WgpuBackend::Metal) => cubecl::Device::metal_msl(kind()),
        Device::Wgpu(WgpuBackend::Vulkan) => cubecl::Device::vulkan(kind()),
        Device::Wgpu(WgpuBackend::Dx12) => cubecl::Device::dx12(kind()),
    };
    let named = named.map_err(|unavailable| failure(device, operation, unavailable))?;
    std::panic::catch_unwind(|| named.client()).map_err(|panic| {
        let problem = panic
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| panic.downcast_ref::<&str>().map(|text| (*text).to_owned()))
            .unwrap_or_else(|| "the runtime panicked while starting".to_owned());
        failure(device, operation, problem)
    })
}

#[cfg(any(feature = "wgpu", feature = "cuda"))]
mod run {
    //! The launches, only in a build with a GPU runtime.

    use cubecl::prelude::*;
    use cubecl::server::Handle;
    use kyto_error::KytoError;

    use super::{DeviceKernels, RowValues, failure};
    use crate::kernels as cube;

    /// Threads per cube (workgroup): 256 runs on every runtime kyto supports.
    const CUBE_THREADS: u32 = 256;

    /// The most cubes along one axis of a dispatch (wgpu's limit, 65535).
    const MAX_CUBES_PER_AXIS: u32 = 65_535;

    /// A dispatch of at least `threads` threads: cubes along x, then y when x is full.
    /// `ABSOLUTE_POS` counts across both, and each kernel checks its bound.
    fn cubes(threads: usize) -> CubeCount {
        let cubes = threads.div_ceil(CUBE_THREADS as usize) as u64;
        let x = cubes.clamp(1, u64::from(MAX_CUBES_PER_AXIS));
        let y = cubes.div_ceil(x).max(1);
        CubeCount::Static(x as u32, y as u32, 1)
    }

    impl DeviceKernels {
        /// Bytes of `values` on the device.
        fn upload<T: CubeElement>(&self, values: &[T]) -> Handle {
            self.client.create_from_slice(T::as_bytes(values))
        }

        /// An output buffer of `len` values of `T`.
        fn output<T>(&self, len: usize) -> Handle {
            self.client.empty(len * std::mem::size_of::<T>())
        }

        /// The values of `handle` back on the host.
        fn download<T: CubeElement>(
            &self,
            handle: Handle,
            operation: &str,
        ) -> Result<Vec<T>, KytoError> {
            let bytes = self
                .client
                .read_one(handle)
                .map_err(|problem| failure(self.device, operation, problem))?;
            Ok(T::from_bytes(&bytes).to_vec())
        }

        /// `indptr` as `u32` offsets from its first entry, on the device.
        ///
        /// # Errors
        /// `KY-E505` if the block holds more than `u32::MAX` values.
        fn upload_indptr(&self, indptr: &[u64], operation: &str) -> Result<Handle, KytoError> {
            let first = indptr.first().copied().unwrap_or(0);
            let offsets = indptr
                .iter()
                .map(|&offset| u32::try_from(offset - first))
                .collect::<Result<Vec<u32>, _>>()
                .map_err(|_| {
                    failure(
                        self.device,
                        operation,
                        "a block holds more than 2^32 values; pass a smaller memory_limit",
                    )
                })?;
            Ok(self.upload(&offsets))
        }
    }

    /// `$body` with `$values` bound to the slice of `$row_values` and `$type` to its
    /// element type.
    macro_rules! with_values {
        ($row_values:expr, |$values:ident: $type:ident| $body:expr) => {
            match $row_values {
                RowValues::F32($values) => {
                    type $type = f32;
                    $body
                }
                RowValues::U32($values) => {
                    type $type = u32;
                    $body
                }
            }
        };
    }

    /// Each row's sum, in `f64`: a `u32` row exactly (as the CPU's, below 2^53), an
    /// `f32` row as a compensated `f32` pair (`kernels::row_sums_f32`), within about
    /// 1e-11 relative of the exact sum (5e-12 measured on Metal).
    ///
    /// # Errors
    /// `KY-E505` if a copy or the kernel fails; `operation` names the plan step.
    pub(super) fn row_sums(
        gpu: &DeviceKernels,
        indptr: &[u64],
        values: RowValues<'_>,
        operation: &str,
    ) -> Result<Vec<f64>, KytoError> {
        let rows = indptr.len().saturating_sub(1);
        if rows == 0 {
            return Ok(Vec::new());
        }
        if values.is_empty() {
            return Ok(vec![0.0; rows]);
        }
        let offsets = gpu.upload_indptr(indptr, operation)?;
        let (high, low) = (gpu.output::<u32>(rows), gpu.output::<u32>(rows));
        let arguments = |handle: &Handle, len| unsafe {
            // SAFETY: each handle was allocated above for `len` elements of the
            // kernel argument's 4-byte type.
            BufferArg::from_raw_parts(handle.clone(), len)
        };
        match values {
            RowValues::U32(counts) => {
                let input = gpu.upload(counts);
                cube::row_sums_u32::launch(
                    &gpu.client,
                    cubes(rows),
                    CubeDim::new_1d(CUBE_THREADS),
                    arguments(&offsets, rows + 1),
                    arguments(&input, counts.len()),
                    arguments(&high, rows),
                    arguments(&low, rows),
                );
                let high = gpu.download::<u32>(high, operation)?;
                let low = gpu.download::<u32>(low, operation)?;
                Ok(high
                    .iter()
                    .zip(&low)
                    .map(|(&high, &low)| ((u64::from(high) << 32) | u64::from(low)) as f64)
                    .collect())
            }
            RowValues::F32(values) => {
                let input = gpu.upload(values);
                cube::row_sums_f32::launch::<f32>(
                    &gpu.client,
                    cubes(rows),
                    CubeDim::new_1d(CUBE_THREADS),
                    arguments(&offsets, rows + 1),
                    arguments(&input, values.len()),
                    arguments(&high, rows),
                    arguments(&low, rows),
                    1.0,
                );
                let high = gpu.download::<f32>(high, operation)?;
                let low = gpu.download::<f32>(low, operation)?;
                Ok(high
                    .iter()
                    .zip(&low)
                    .map(|(&high, &low)| {
                        let high = f64::from(high);
                        if high.is_finite() && low.is_finite() {
                            high + f64::from(low)
                        } else {
                            high
                        }
                    })
                    .collect())
            }
        }
    }

    /// Each row's number of nonzero values (a stored zero is not counted, a NaN is).
    ///
    /// # Errors
    /// `KY-E505` if a copy or the kernel fails.
    pub(super) fn row_nnz(
        gpu: &DeviceKernels,
        indptr: &[u64],
        values: RowValues<'_>,
        operation: &str,
    ) -> Result<Vec<u64>, KytoError> {
        let rows = indptr.len().saturating_sub(1);
        if rows == 0 {
            return Ok(Vec::new());
        }
        if values.is_empty() {
            return Ok(vec![0; rows]);
        }
        let offsets = gpu.upload_indptr(indptr, operation)?;
        let counts = gpu.output::<u32>(rows);
        with_values!(values, |values: Value| {
            let input = gpu.upload(values);
            // SAFETY: each handle holds the stated number of 4-byte elements.
            unsafe {
                cube::row_nnz::launch::<Value>(
                    &gpu.client,
                    cubes(rows),
                    CubeDim::new_1d(CUBE_THREADS),
                    BufferArg::from_raw_parts(offsets, rows + 1),
                    BufferArg::from_raw_parts(input, values.len()),
                    BufferArg::from_raw_parts(counts.clone(), rows),
                );
            }
        });
        Ok(gpu
            .download::<u32>(counts, operation)?
            .into_iter()
            .map(u64::from)
            .collect())
    }

    /// Each value divided by its row's divisor in `f32`, written to `out`.
    ///
    /// # Errors
    /// `KY-E505` if a copy or the kernel fails.
    ///
    /// # Panics
    /// If `divisors` has not one divisor per row or `out` is not as long as
    /// `values` (as the CPU kernel).
    pub(super) fn divide_rows(
        gpu: &DeviceKernels,
        indptr: &[u64],
        values: RowValues<'_>,
        divisors: &[f32],
        out: &mut [f32],
        operation: &str,
    ) -> Result<(), KytoError> {
        let rows = indptr.len().saturating_sub(1);
        assert_eq!(divisors.len(), rows, "one divisor per row");
        assert_eq!(out.len(), values.len(), "out as long as values");
        if values.is_empty() {
            return Ok(());
        }
        let mut row_of = Vec::with_capacity(values.len());
        for (row, bounds) in indptr.windows(2).enumerate() {
            let row = u32::try_from(row)
                .map_err(|_| failure(gpu.device, operation, "a block holds more than 2^32 rows"))?;
            row_of.extend(std::iter::repeat_n(row, (bounds[1] - bounds[0]) as usize));
        }
        let row_of = gpu.upload(&row_of);
        let divisors = gpu.upload(divisors);
        let output = gpu.output::<f32>(out.len());
        with_values!(values, |values: Value| {
            let input = gpu.upload(values);
            // SAFETY: each handle holds the stated number of 4-byte elements.
            unsafe {
                cube::divide_rows::launch::<Value>(
                    &gpu.client,
                    cubes(values.len()),
                    CubeDim::new_1d(CUBE_THREADS),
                    BufferArg::from_raw_parts(row_of, values.len()),
                    BufferArg::from_raw_parts(input, values.len()),
                    BufferArg::from_raw_parts(divisors, rows),
                    BufferArg::from_raw_parts(output.clone(), values.len()),
                );
            }
        });
        out.copy_from_slice(&gpu.download::<f32>(output, operation)?);
        Ok(())
    }

    /// `log1p` of each value (a `u32` rounded to `f32` first), written to `out`.
    ///
    /// # Errors
    /// `KY-E505` if a copy or the kernel fails.
    ///
    /// # Panics
    /// If `out` is not as long as `values`.
    pub(super) fn log1p(
        gpu: &DeviceKernels,
        values: RowValues<'_>,
        out: &mut [f32],
        operation: &str,
    ) -> Result<(), KytoError> {
        assert_eq!(out.len(), values.len(), "out as long as values");
        if values.is_empty() {
            return Ok(());
        }
        let output = gpu.output::<f32>(out.len());
        with_values!(values, |values: Value| {
            let input = gpu.upload(values);
            // SAFETY: each handle holds the stated number of 4-byte elements.
            unsafe {
                cube::log1p::launch::<Value>(
                    &gpu.client,
                    cubes(values.len()),
                    CubeDim::new_1d(CUBE_THREADS),
                    BufferArg::from_raw_parts(input, values.len()),
                    BufferArg::from_raw_parts(output.clone(), values.len()),
                );
            }
        });
        out.copy_from_slice(&gpu.download::<f32>(output, operation)?);
        Ok(())
    }
}
