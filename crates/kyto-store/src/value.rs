//! Value dtypes a layer may store (format.md §4.2): `float32` and `uint32`.
//!
//! Does: map each Rust value type to its Zarr dtype and say which values count as
//! zero, so writers drop them (format.md §4.6).
//! Does not: convert between dtypes; a layer is read back as the dtype it stores.

use zarrs::array::{DataType, Element, ElementOwned, FillValue};

/// A value type a layer can store: `f32` (`float32`) or `u32` (`uint32`).
///
/// The trait is sealed: v0 allows no other dtype (format.md §4.2).
pub trait LayerValue:
    Element + ElementOwned + bytemuck::Pod + PartialEq + std::fmt::Debug + Send + Sync + sealed::Sealed
{
    /// The Zarr v3 dtype name of the layer's `data` array.
    const ZARR_DTYPE: &'static str;

    /// The Zarr data type of the layer's `data` array.
    fn data_type() -> DataType;

    /// The value used as the Zarr `fill_value` and for padding (0).
    fn zero() -> Self;

    /// The Zarr `fill_value` of the layer's `data` array (0, format.md §4.2).
    fn fill_value() -> FillValue;

    /// True when the value counts as zero and must not be stored (format.md §4.6).
    /// For `f32`, `-0.0` counts as zero; `NaN` and infinities do not.
    fn is_zero(self) -> bool;
}

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for u32 {}
}

impl LayerValue for f32 {
    const ZARR_DTYPE: &'static str = "float32";

    fn data_type() -> DataType {
        DataType::Float32
    }

    fn zero() -> Self {
        0.0
    }

    fn fill_value() -> FillValue {
        FillValue::from(0.0_f32)
    }

    fn is_zero(self) -> bool {
        // `-0.0 == 0.0` is true and `NaN == 0.0` is false, as format.md §4.6 requires.
        self == 0.0
    }
}

impl LayerValue for u32 {
    const ZARR_DTYPE: &'static str = "uint32";

    fn data_type() -> DataType {
        DataType::UInt32
    }

    fn zero() -> Self {
        0
    }

    fn fill_value() -> FillValue {
        FillValue::from(0_u32)
    }

    fn is_zero(self) -> bool {
        self == 0
    }
}

#[cfg(test)]
mod tests {
    use super::LayerValue;

    #[test]
    fn zero_rules_follow_format_4_6() {
        assert!(0.0_f32.is_zero());
        assert!((-0.0_f32).is_zero());
        assert!(!f32::NAN.is_zero());
        assert!(!f32::INFINITY.is_zero());
        assert!(!f32::MIN_POSITIVE.is_zero());
        assert!(0_u32.is_zero());
        assert!(!1_u32.is_zero());
    }
}
