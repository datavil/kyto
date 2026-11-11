//! Open one tiled sparse layer and check its cheap invariants (format.md §1.2, §4.2).
//!
//! Does: check the layer name and group kind, open the four arrays on any readable
//! Zarr storage, check their dtypes and shapes against the tile grid, and read and
//! check `tile_offsets` (format.md §4.7 step 1).
//! Does not: read `indptr`, `indices` or `data` (see `layer_io` for a whole layer,
//! `tile_reader` for selected tiles), or check `format_version` (see `store_version`).

use std::sync::Arc;

use kyto_core::LayerDtype;
use kyto_error::KytoError;
use zarrs::array::{Array, DataType, ElementOwned};
use zarrs::group::Group;
use zarrs::storage::{ReadableStorageTraits, StoreKey};

use crate::names::validate_member_name;
use crate::tile_decode::check_tile_offsets;
use crate::tiling::TileGrid;
use crate::value::LayerValue;
use crate::zarr_arrays::store_io;

/// Value of the layer group's `attributes.kyto.kind` (format.md §4.2).
pub(crate) const LAYER_KIND: &str = "tiled_csr";

/// The value dtype a layer stores (format.md §4.2).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueDtype {
    /// `float32`.
    F32,
    /// `uint32`.
    U32,
}

impl From<ValueDtype> for LayerDtype {
    fn from(dtype: ValueDtype) -> Self {
        match dtype {
            ValueDtype::F32 => Self::Float32,
            ValueDtype::U32 => Self::Uint32,
        }
    }
}

impl From<LayerDtype> for ValueDtype {
    fn from(dtype: LayerDtype) -> Self {
        match dtype {
            LayerDtype::Float32 => Self::F32,
            LayerDtype::Uint32 => Self::U32,
        }
    }
}

impl ValueDtype {
    /// The Zarr v3 dtype name, as `zarr.json` and `schema.json` write it.
    pub fn zarr_name(self) -> &'static str {
        match self {
            Self::F32 => f32::ZARR_DTYPE,
            Self::U32 => u32::ZARR_DTYPE,
        }
    }
}

/// An opened layer whose arrays passed the checks of [`open_layer`].
pub(crate) struct OpenedLayer<S: ?Sized> {
    /// The layer name.
    pub name: String,
    /// `indptr`, `uint32 [n_cell_blocks, n_gene_blocks, C + 1]`.
    pub indptr: Array<S>,
    /// `indices`, `uint32 [nnz]`.
    pub indices: Array<S>,
    /// `data`, `float32` or `uint32 [nnz]`.
    pub data: Array<S>,
    /// The dtype of `data`.
    pub dtype: ValueDtype,
    /// `tile_offsets`, read whole and checked (`n_tiles + 1` values ending at nnz).
    pub tile_offsets: Vec<u64>,
}

/// Open layer `name` of the store in `storage`, laid out on `grid`.
///
/// `store` names the store in error messages.
///
/// # Errors
/// - `KY-E102` if `name` breaks format.md §2.1.
/// - `KY-E107` if the group or an array breaks format.md §4.2 or `tile_offsets` breaks
///   §4.3.
/// - `KY-E109` if the store has no layer `name`.
/// - `KY-E110` if the storage fails.
pub(crate) fn open_layer<S: ?Sized + ReadableStorageTraits + 'static>(
    storage: &Arc<S>,
    store: &str,
    name: &str,
    grid: &TileGrid,
) -> Result<OpenedLayer<S>, KytoError> {
    validate_member_name("layer", name)?;
    let group_path = format!("/layers/{name}");
    let group_key = StoreKey::new(format!("layers/{name}/zarr.json"))
        .map_err(|error| store_io("open", &group_path, error))?;
    let exists = storage
        .size_key(&group_key)
        .map_err(|error| store_io("open", &group_path, error))?;
    if exists.is_none() {
        // The storage layer holds no list of layers to suggest from; the plan checks
        // layer names first, with suggestions (task 8.2).
        return Err(KytoError::MissingMember {
            kind: "layer".to_owned(),
            name: name.to_owned(),
            store: store.to_owned(),
            suggestions: Box::default(),
        });
    }
    let corrupt = |problem: String| KytoError::CorruptLayer {
        layer: name.to_owned(),
        problem,
    };
    let group = Group::open(storage.clone(), &group_path)
        .map_err(|error| corrupt(format!("cannot open the layer group: {error}")))?;
    let kind = group
        .attributes()
        .get("kyto")
        .and_then(|kyto| kyto.get("kind"))
        .and_then(|kind| kind.as_str());
    if kind != Some(LAYER_KIND) {
        return Err(corrupt(format!(
            "attributes.kyto.kind is {kind:?}, not \"{LAYER_KIND}\""
        )));
    }
    let open = |array: &str, data_type: Option<DataType>, shape: Option<Vec<u64>>| {
        let opened = Array::open(storage.clone(), &format!("{group_path}/{array}"))
            .map_err(|error| corrupt(format!("cannot open array {array}: {error}")))?;
        if let Some(expected) = data_type
            && opened.data_type().to_string() != expected.to_string()
        {
            return Err(corrupt(format!(
                "array {array} has dtype {}, not {expected}",
                opened.data_type()
            )));
        }
        let found = opened.shape();
        let shape_ok = match &shape {
            Some(expected) => found == expected.as_slice(),
            None => found.len() == 1,
        };
        if !shape_ok {
            return Err(corrupt(format!(
                "array {array} has shape {found:?}, expected {}",
                shape.map_or("one dimension".to_owned(), |expected| format!(
                    "{expected:?}"
                ))
            )));
        }
        Ok(opened)
    };
    let indptr_shape = vec![
        grid.n_cell_blocks(),
        grid.n_gene_blocks(),
        grid.indptr_len(),
    ];
    let indptr = open("indptr", Some(DataType::UInt32), Some(indptr_shape))?;
    let tile_offsets = open(
        "tile_offsets",
        Some(DataType::UInt64),
        Some(vec![grid.n_tiles() + 1]),
    )?;
    let indices = open("indices", Some(DataType::UInt32), None)?;
    let data = open("data", None, Some(indices.shape().to_vec()))?;
    let stored = data.data_type().to_string();
    let dtype = if stored == f32::ZARR_DTYPE {
        ValueDtype::F32
    } else if stored == u32::ZARR_DTYPE {
        ValueDtype::U32
    } else {
        return Err(corrupt(format!(
            "array data has dtype {stored}; a layer stores float32 or uint32"
        )));
    };

    // Check the entry count before reading entries, so a bad zarr.json cannot make a
    // reader allocate more than a dense matrix of the given shape.
    let nnz = indices.shape().first().copied().unwrap_or(0);
    if u128::from(nnz) > u128::from(grid.n_cells()) * u128::from(grid.n_genes()) {
        return Err(corrupt(format!(
            "indices holds {nnz} entries, more than n_cells * n_genes"
        )));
    }
    let tile_offsets: Vec<u64> = read_all(&tile_offsets)?;
    check_tile_offsets(name, grid, &tile_offsets, nnz)?;
    Ok(OpenedLayer {
        name: name.to_owned(),
        indptr,
        indices,
        data,
        dtype,
        tile_offsets,
    })
}

/// Read a whole array.
pub(crate) fn read_all<S, T>(array: &Array<S>) -> Result<Vec<T>, KytoError>
where
    S: ?Sized + ReadableStorageTraits + 'static,
    T: ElementOwned,
{
    array
        .retrieve_array_subset_elements(&array.subset_all())
        .map_err(|error| store_io("read", array.path().as_str(), error))
}
