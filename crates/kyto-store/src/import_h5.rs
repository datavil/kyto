//! HDF5 access for the `.h5ad` and 10x `.h5` importers (task 2.6).
//!
//! Does: open files, read string attributes, string datasets (variable or fixed
//! length, in chunks), numeric slices as `f64` or `i64`, and 1-d columns as Polars
//! series of their own dtype; turn HDF5 failures into coded errors naming the member.
//! Does not: know the anndata or 10x layouts (see `import_h5ad`, `import_10x`).
//!
//! HDF5 is the static library bundled by `hdf5-metno` (user decision, Nov 7, 2026;
//! memory/hdf5_static_build.md).

use std::ops::Range;
use std::path::Path;

use hdf5::types::{FixedAscii, FixedUnicode, TypeDescriptor, VarLenAscii, VarLenUnicode};
use hdf5::{Dataset, File, Group, H5Type, Location};
use hdf5_metno as hdf5;
use kyto_error::KytoError;
use polars::prelude::{NamedFrom, Series};

/// Elements read per call for strings and scans: bounds the read buffers.
pub(crate) const READ_CHUNK: usize = 1 << 20;

/// Longest fixed-length string read (bytes).
const MAX_FIXED_STRING: usize = 4096;

/// An open HDF5 input and the path it was opened from, for messages.
pub(crate) struct H5File {
    pub file: File,
    pub path: String,
}

impl H5File {
    /// Open `path` read-only.
    ///
    /// # Errors
    /// `KY-E110` if it is missing or cannot be read, `KY-E120` if it is not HDF5.
    pub(crate) fn open(path: &Path) -> Result<Self, KytoError> {
        let text = path.display().to_string();
        if let Err(error) = std::fs::metadata(path) {
            return Err(KytoError::StoreIo {
                operation: "open input".to_owned(),
                path: text,
                detail: error.to_string(),
            });
        }
        let file = File::open(path).map_err(|error| KytoError::UnsupportedInput {
            path: text.clone(),
            problem: format!("it is not a readable HDF5 file ({error})"),
            accepted:
                "pass an .h5ad file written by anndata or a 10x .h5 file written by Cell Ranger"
                    .to_owned(),
        })?;
        Ok(Self { file, path: text })
    }

    /// `path:/member`, the label of a member in messages.
    pub(crate) fn label(&self, member: &str) -> String {
        format!("{}:/{}", self.path, member.trim_start_matches('/'))
    }

    /// Whether `member` exists (any kind).
    pub(crate) fn exists(&self, member: &str) -> bool {
        self.file.link_exists(member)
    }

    /// The group at `member`.
    ///
    /// # Errors
    /// `KY-E120` if it is missing or not a group.
    pub(crate) fn group(&self, member: &str, accepted: &str) -> Result<Group, KytoError> {
        self.file
            .group(member)
            .map_err(|_| self.unsupported(member, "it is missing or not a group", accepted))
    }

    /// The dataset at `member`.
    ///
    /// # Errors
    /// `KY-E120` if it is missing or not a dataset.
    pub(crate) fn dataset(&self, member: &str, accepted: &str) -> Result<Dataset, KytoError> {
        self.file
            .dataset(member)
            .map_err(|_| self.unsupported(member, "it is missing or not a dataset", accepted))
    }

    /// A `KY-E120` error about `member`.
    pub(crate) fn unsupported(&self, member: &str, problem: &str, accepted: &str) -> KytoError {
        KytoError::UnsupportedInput {
            path: self.label(member),
            problem: problem.to_owned(),
            accepted: accepted.to_owned(),
        }
    }

    /// A `KY-E110` error for a failed read of `member`.
    pub(crate) fn read_error(&self, member: &str, error: impl std::fmt::Display) -> KytoError {
        KytoError::StoreIo {
            operation: "read input".to_owned(),
            path: self.label(member),
            detail: error.to_string(),
        }
    }
}

/// A string attribute (variable or fixed length), or `None` if absent or not a
/// scalar string.
pub(crate) fn attr_string(location: &Location, name: &str) -> Option<String> {
    let attribute = location.attr(name).ok()?;
    if !attribute.is_scalar() {
        return None;
    }
    match attribute.dtype().ok()?.to_descriptor().ok()? {
        TypeDescriptor::VarLenUnicode => attribute
            .read_scalar::<VarLenUnicode>()
            .ok()
            .and_then(|text| utf8(text.as_bytes())),
        TypeDescriptor::VarLenAscii => attribute
            .read_scalar::<VarLenAscii>()
            .ok()
            .and_then(|text| utf8(text.as_bytes())),
        TypeDescriptor::FixedAscii(size) | TypeDescriptor::FixedUnicode(size)
            if size <= MAX_FIXED_STRING =>
        {
            attribute
                .read_scalar::<FixedUnicode<MAX_FIXED_STRING>>()
                .ok()
                .and_then(|text| String::from_utf8(text.as_bytes().to_vec()).ok())
        }
        _ => None,
    }
}

/// A 1-d string array attribute; an empty array of any dtype gives no strings
/// (h5py writes an empty `column-order` as `float64`). `None` if absent or not strings.
pub(crate) fn attr_strings(location: &Location, name: &str) -> Option<Vec<String>> {
    let attribute = location.attr(name).ok()?;
    if attribute.is_scalar() {
        return attr_string(location, name).map(|text| vec![text]);
    }
    if attribute.size() == 0 {
        return Some(Vec::new());
    }
    match attribute.dtype().ok()?.to_descriptor().ok()? {
        TypeDescriptor::VarLenUnicode => attribute
            .read_raw::<VarLenUnicode>()
            .ok()?
            .iter()
            .map(|text| utf8(text.as_bytes()))
            .collect(),
        TypeDescriptor::VarLenAscii => attribute
            .read_raw::<VarLenAscii>()
            .ok()?
            .iter()
            .map(|text| utf8(text.as_bytes()))
            .collect(),
        _ => None,
    }
}

/// Bytes as a string if they are UTF-8 (hdf5-metno's `as_str` does not check).
fn utf8(bytes: &[u8]) -> Option<String> {
    std::str::from_utf8(bytes).ok().map(str::to_owned)
}

/// A boolean attribute (`numpy.bool_` is an HDF5 enum), or `None`.
pub(crate) fn attr_bool(location: &Location, name: &str) -> Option<bool> {
    location.attr(name).ok()?.read_scalar::<bool>().ok()
}

/// A 1-d integer attribute such as `shape`, or `None`.
pub(crate) fn attr_u64s(location: &Location, name: &str) -> Option<Vec<u64>> {
    location.attr(name).ok()?.read_raw::<u64>().ok()
}

/// The 1-d length of `dataset`, or `KY-E120` if it is not 1-d.
pub(crate) fn length_1d(file: &H5File, member: &str, dataset: &Dataset) -> Result<u64, KytoError> {
    match dataset.shape().as_slice() {
        [length] => Ok(*length as u64),
        shape => Err(file.unsupported(
            member,
            &format!("it has shape {shape:?}, not one dimension"),
            "store one value per cell or gene",
        )),
    }
}

/// The type descriptor of `dataset`.
pub(crate) fn descriptor(
    file: &H5File,
    member: &str,
    dataset: &Dataset,
) -> Result<TypeDescriptor, KytoError> {
    dataset
        .dtype()
        .and_then(|dtype| dtype.to_descriptor())
        .map_err(|error| {
            file.unsupported(
                member,
                &format!("its dtype is not supported ({error})"),
                "store values as booleans, integers, float32, float64 or strings",
            )
        })
}

/// Whether `dataset` holds `float64` values.
pub(crate) fn is_float64(
    file: &H5File,
    member: &str,
    dataset: &Dataset,
) -> Result<bool, KytoError> {
    Ok(matches!(
        descriptor(file, member, dataset)?,
        TypeDescriptor::Float(hdf5::types::FloatSize::U8)
    ))
}

/// Every string of a 1-d string dataset, read in chunks.
///
/// # Errors
/// `KY-E120` if it does not hold strings or a fixed length is above 4096 bytes,
/// `KY-E121` if a fixed-length string is not UTF-8, `KY-E110` if a read fails.
pub(crate) fn read_strings(
    file: &H5File,
    member: &str,
    dataset: &Dataset,
) -> Result<Vec<String>, KytoError> {
    let length = length_1d(file, member, dataset)? as usize;
    let accepted = "store strings as HDF5 strings (h5py writes these)";
    let mut strings = Vec::with_capacity(length);
    match descriptor(file, member, dataset)? {
        TypeDescriptor::VarLenUnicode => {
            read_chunks::<VarLenUnicode>(file, member, dataset, length, &mut |texts| {
                for text in texts {
                    let position = strings.len();
                    strings.push(decode_utf8(file, member, text.as_bytes(), position)?);
                }
                Ok(())
            })?;
        }
        TypeDescriptor::VarLenAscii => {
            read_chunks::<VarLenAscii>(file, member, dataset, length, &mut |texts| {
                for text in texts {
                    let position = strings.len();
                    strings.push(decode_utf8(file, member, text.as_bytes(), position)?);
                }
                Ok(())
            })?;
        }
        TypeDescriptor::FixedAscii(size) => {
            read_fixed::<FixedAscii<64>, FixedAscii<MAX_FIXED_STRING>>(
                file,
                member,
                dataset,
                length,
                size,
                &mut strings,
                |text| text.as_bytes(),
                |text| text.as_bytes(),
            )?;
        }
        TypeDescriptor::FixedUnicode(size) => {
            read_fixed::<FixedUnicode<64>, FixedUnicode<MAX_FIXED_STRING>>(
                file,
                member,
                dataset,
                length,
                size,
                &mut strings,
                |text| text.as_bytes(),
                |text| text.as_bytes(),
            )?;
        }
        other => {
            return Err(file.unsupported(
                member,
                &format!("it holds {other} values, not strings"),
                accepted,
            ));
        }
    }
    Ok(strings)
}

/// String `position` of `member` from its bytes, or `KY-E121` if not UTF-8.
fn decode_utf8(
    file: &H5File,
    member: &str,
    bytes: &[u8],
    position: usize,
) -> Result<String, KytoError> {
    String::from_utf8(bytes.to_vec()).map_err(|_| KytoError::InvalidInput {
        path: file.label(member),
        location: format!("element {position}"),
        problem: "the string is not valid UTF-8".to_owned(),
        advice: "write strings as UTF-8".to_owned(),
    })
}

/// Fixed-length strings, read through the short type when they fit it (less memory).
#[allow(clippy::too_many_arguments)]
fn read_fixed<Short: H5Type, Long: H5Type>(
    file: &H5File,
    member: &str,
    dataset: &Dataset,
    length: usize,
    size: usize,
    strings: &mut Vec<String>,
    short_bytes: fn(&Short) -> &[u8],
    long_bytes: fn(&Long) -> &[u8],
) -> Result<(), KytoError> {
    let decode = |bytes: &[u8], position: usize| decode_utf8(file, member, bytes, position);
    if size <= 64 {
        read_chunks::<Short>(file, member, dataset, length, &mut |texts| {
            for text in texts {
                let position = strings.len();
                strings.push(decode(short_bytes(text), position)?);
            }
            Ok(())
        })
    } else if size <= MAX_FIXED_STRING {
        // 4096 elements of 4096 bytes: a 16 MiB buffer.
        read_chunks_sized::<Long>(file, member, dataset, length, 1 << 12, &mut |texts| {
            for text in texts {
                let position = strings.len();
                strings.push(decode(long_bytes(text), position)?);
            }
            Ok(())
        })
    } else {
        Err(file.unsupported(
            member,
            &format!("its strings are {size} bytes long, above the {MAX_FIXED_STRING}-byte limit"),
            "store long strings as variable-length strings",
        ))
    }
}

/// Read `0..length` of a 1-d dataset in chunks of [`READ_CHUNK`] elements.
fn read_chunks<T: H5Type>(
    file: &H5File,
    member: &str,
    dataset: &Dataset,
    length: usize,
    visit: &mut dyn FnMut(&[T]) -> Result<(), KytoError>,
) -> Result<(), KytoError> {
    read_chunks_sized(file, member, dataset, length, READ_CHUNK, visit)
}

fn read_chunks_sized<T: H5Type>(
    file: &H5File,
    member: &str,
    dataset: &Dataset,
    length: usize,
    chunk: usize,
    visit: &mut dyn FnMut(&[T]) -> Result<(), KytoError>,
) -> Result<(), KytoError> {
    let mut start = 0;
    while start < length {
        let end = (start + chunk).min(length);
        let values = read_slice::<T>(file, member, dataset, start..end)?;
        visit(&values)?;
        start = end;
    }
    Ok(())
}

/// Elements `range` of a 1-d dataset, converted to `T` by HDF5 (empty ranges read
/// nothing).
pub(crate) fn read_slice<T: H5Type>(
    file: &H5File,
    member: &str,
    dataset: &Dataset,
    range: Range<usize>,
) -> Result<Vec<T>, KytoError> {
    if range.is_empty() {
        return Ok(Vec::new());
    }
    dataset
        .read_slice_1d::<T, _>(range)
        .map(|values| values.into_raw_vec_and_offset().0)
        .map_err(|error| file.read_error(member, error))
}

/// Every element of a 1-d dataset as `T`.
pub(crate) fn read_all<T: H5Type>(
    file: &H5File,
    member: &str,
    dataset: &Dataset,
) -> Result<Vec<T>, KytoError> {
    let length = length_1d(file, member, dataset)? as usize;
    read_slice(file, member, dataset, 0..length)
}

/// A 1-d numeric or boolean dataset as a series of its own dtype; `missing[i]` true
/// makes row `i` null (anndata's nullable arrays).
///
/// # Errors
/// `KY-E120` for other dtypes (strings, compounds, `float16`), `KY-E110` if a read
/// fails.
pub(crate) fn numeric_series(
    file: &H5File,
    member: &str,
    name: &str,
    dataset: &Dataset,
    missing: Option<&[bool]>,
) -> Result<Series, KytoError> {
    use hdf5::types::{FloatSize, IntSize};
    let descriptor = descriptor(file, member, dataset)?;
    Ok(match descriptor {
        TypeDescriptor::Boolean => typed::<bool>(file, member, name, dataset, missing)?,
        TypeDescriptor::Integer(IntSize::U1) => typed::<i8>(file, member, name, dataset, missing)?,
        TypeDescriptor::Integer(IntSize::U2) => typed::<i16>(file, member, name, dataset, missing)?,
        TypeDescriptor::Integer(IntSize::U4) => typed::<i32>(file, member, name, dataset, missing)?,
        TypeDescriptor::Integer(IntSize::U8) => typed::<i64>(file, member, name, dataset, missing)?,
        TypeDescriptor::Unsigned(IntSize::U1) => typed::<u8>(file, member, name, dataset, missing)?,
        TypeDescriptor::Unsigned(IntSize::U2) => {
            typed::<u16>(file, member, name, dataset, missing)?
        }
        TypeDescriptor::Unsigned(IntSize::U4) => {
            typed::<u32>(file, member, name, dataset, missing)?
        }
        TypeDescriptor::Unsigned(IntSize::U8) => {
            typed::<u64>(file, member, name, dataset, missing)?
        }
        TypeDescriptor::Float(FloatSize::U4) => typed::<f32>(file, member, name, dataset, missing)?,
        TypeDescriptor::Float(FloatSize::U8) => typed::<f64>(file, member, name, dataset, missing)?,
        other => {
            return Err(file.unsupported(
                member,
                &format!("it holds {other} values"),
                "store numeric columns as booleans, integers, float32 or float64",
            ));
        }
    })
}

fn typed<T>(
    file: &H5File,
    member: &str,
    name: &str,
    dataset: &Dataset,
    missing: Option<&[bool]>,
) -> Result<Series, KytoError>
where
    T: H5Type + Copy,
    Series: NamedFrom<Vec<T>, [T]> + NamedFrom<Vec<Option<T>>, [Option<T>]>,
{
    let values = read_all::<T>(file, member, dataset)?;
    Ok(match missing {
        None => Series::new(name.into(), values),
        Some(missing) => Series::new(
            name.into(),
            values
                .iter()
                .zip(missing)
                .map(|(&value, &is_missing)| (!is_missing).then_some(value))
                .collect::<Vec<Option<T>>>(),
        ),
    })
}

/// The layer dtype of a numeric matrix dataset: integers become `uint32`, floats
/// `float32` (format.md §4.2, deviations.md).
///
/// # Errors
/// `KY-E120` for booleans, strings and other dtypes.
pub(crate) fn matrix_dtype(
    file: &H5File,
    member: &str,
    dataset: &Dataset,
) -> Result<crate::layer_open::ValueDtype, KytoError> {
    use crate::layer_open::ValueDtype;
    match descriptor(file, member, dataset)? {
        TypeDescriptor::Integer(_) | TypeDescriptor::Unsigned(_) => Ok(ValueDtype::U32),
        TypeDescriptor::Float(_) => Ok(ValueDtype::F32),
        other => Err(file.unsupported(
            member,
            &format!("matrix values are {other}"),
            "store matrix values as integers (imported as uint32) or floats (imported as float32)",
        )),
    }
}
