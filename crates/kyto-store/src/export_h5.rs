//! HDF5 writing helpers for exports (task 2.7): groups, anndata encoding attributes,
//! string and numeric datasets, each failure as a coded error naming the member.
//!
//! Does: wrap an `hdf5-metno` file being written with the target path for messages, and
//! write the element kinds anndata's h5py writer produces: variable-length UTF-8
//! strings, numeric and boolean (h5py's `int8` enum) arrays, scalar and array
//! attributes.
//! Does not: know the anndata layout (see `export_h5ad`, `export_h5ad_table`).

use std::path::Path;
use std::str::FromStr;

use hdf5_metno::types::VarLenUnicode;
use hdf5_metno::{Dataset, File, Group, H5Type, Location};
use kyto_error::KytoError;

/// An HDF5 file being written, and the target path its messages name.
pub(crate) struct H5Out {
    pub file: File,
    /// The final target path (not the temporary file), for messages.
    pub target: String,
}

impl H5Out {
    /// Create (truncate) the HDF5 file at `path`; messages name `target`.
    ///
    /// # Errors
    /// `KY-E110` if it cannot be created.
    pub(crate) fn create(path: &Path, target: &Path) -> Result<Self, KytoError> {
        let target = target.display().to_string();
        let file = File::create(path).map_err(|error| KytoError::StoreIo {
            operation: "create".to_owned(),
            path: target.clone(),
            detail: error.to_string(),
        })?;
        Ok(Self { file, target })
    }

    /// `KY-E110` for a failed write of `member`.
    pub(crate) fn write_error(&self, member: &str, error: impl std::fmt::Display) -> KytoError {
        KytoError::StoreIo {
            operation: format!("write member '/{member}' of"),
            path: self.target.clone(),
            detail: error.to_string(),
        }
    }

    /// `KY-E124` for a value or name the format cannot hold.
    pub(crate) fn unsupported(&self, location: &str, problem: &str, advice: &str) -> KytoError {
        KytoError::UnsupportedExport {
            location: location.to_owned(),
            path: self.target.clone(),
            problem: problem.to_owned(),
            advice: advice.to_owned(),
        }
    }

    /// Create group `member` (a full path from the root) with an anndata encoding.
    pub(crate) fn group(&self, member: &str, encoding: (&str, &str)) -> Result<Group, KytoError> {
        let group = self
            .file
            .create_group(member)
            .map_err(|error| self.write_error(member, error))?;
        self.encode(member, &group, encoding)?;
        Ok(group)
    }

    /// Write the `encoding-type` and `encoding-version` attributes of `member`.
    pub(crate) fn encode(
        &self,
        member: &str,
        location: &Location,
        (kind, version): (&str, &str),
    ) -> Result<(), KytoError> {
        self.attr_str(member, location, "encoding-type", kind)?;
        self.attr_str(member, location, "encoding-version", version)
    }

    /// A UTF-8 string, as h5py stores a Python `str`.
    pub(crate) fn text(&self, location: &str, text: &str) -> Result<VarLenUnicode, KytoError> {
        VarLenUnicode::from_str(text).map_err(|_| {
            self.unsupported(
                location,
                "HDF5 strings cannot hold the NUL character",
                "remove NUL characters from the value",
            )
        })
    }

    /// A scalar string attribute.
    pub(crate) fn attr_str(
        &self,
        member: &str,
        location: &Location,
        name: &str,
        value: &str,
    ) -> Result<(), KytoError> {
        let value = self.text(&format!("attribute '{name}' of '/{member}'"), value)?;
        location
            .new_attr::<VarLenUnicode>()
            .shape(())
            .create(name)
            .and_then(|attribute| attribute.as_writer().write_scalar(&value))
            .map_err(|error| self.write_error(member, error))
    }

    /// A 1-d string array attribute; empty lists are written as an empty `float64`
    /// array, as h5py writes an empty Python list (anndata's empty `column-order`).
    pub(crate) fn attr_strings(
        &self,
        member: &str,
        location: &Location,
        name: &str,
        values: &[&str],
    ) -> Result<(), KytoError> {
        let result = if values.is_empty() {
            location.new_attr::<f64>().shape(0).create(name).map(|_| ())
        } else {
            let texts = values
                .iter()
                .map(|value| self.text(&format!("column name '{value}'"), value))
                .collect::<Result<Vec<_>, KytoError>>()?;
            location
                .new_attr_builder()
                .with_data(texts.as_slice())
                .create(name)
                .map(|_| ())
        };
        result.map_err(|error| self.write_error(member, error))
    }

    /// A scalar attribute of an `H5Type` (`bool` as h5py's enum).
    pub(crate) fn attr_scalar<T: H5Type>(
        &self,
        member: &str,
        location: &Location,
        name: &str,
        value: &T,
    ) -> Result<(), KytoError> {
        location
            .new_attr::<T>()
            .shape(())
            .create(name)
            .and_then(|attribute| attribute.as_writer().write_scalar(value))
            .map_err(|error| self.write_error(member, error))
    }

    /// A 1-d attribute of an `H5Type`.
    pub(crate) fn attr_array<T: H5Type>(
        &self,
        member: &str,
        location: &Location,
        name: &str,
        values: &[T],
    ) -> Result<(), KytoError> {
        location
            .new_attr_builder()
            .with_data(values)
            .create(name)
            .map(|_| ())
            .map_err(|error| self.write_error(member, error))
    }

    /// A 1-d dataset `member` holding `values`, with an anndata encoding if given.
    pub(crate) fn dataset<T: H5Type>(
        &self,
        member: &str,
        values: &[T],
        encoding: Option<(&str, &str)>,
    ) -> Result<Dataset, KytoError> {
        let dataset = self
            .file
            .new_dataset_builder()
            .with_data(values)
            .create(member)
            .map_err(|error| self.write_error(member, error))?;
        if let Some(encoding) = encoding {
            self.encode(member, &dataset, encoding)?;
        }
        Ok(dataset)
    }

    /// An empty 1-d dataset `member` of `length` values, filled later with
    /// [`H5Out::write_at`].
    pub(crate) fn empty_dataset<T: H5Type>(
        &self,
        member: &str,
        length: u64,
    ) -> Result<Dataset, KytoError> {
        let length = usize::try_from(length).map_err(|error| self.write_error(member, error))?;
        self.file
            .new_dataset::<T>()
            .shape(length)
            .create(member)
            .map_err(|error| self.write_error(member, error))
    }

    /// Write `values` into `dataset` (`member`) at `offset`.
    pub(crate) fn write_at<T: H5Type>(
        &self,
        member: &str,
        dataset: &Dataset,
        offset: u64,
        values: &[T],
    ) -> Result<(), KytoError> {
        if values.is_empty() {
            return Ok(());
        }
        let start = usize::try_from(offset).map_err(|error| self.write_error(member, error))?;
        dataset
            .as_writer()
            .write_slice(values, start..start + values.len())
            .map_err(|error| self.write_error(member, error))
    }

    /// A string-array dataset (`string-array` 0.2.0) of UTF-8 strings.
    pub(crate) fn strings<'a>(
        &self,
        member: &str,
        values: impl Iterator<Item = &'a str>,
        location: &dyn Fn(usize) -> String,
    ) -> Result<Dataset, KytoError> {
        let texts = values
            .enumerate()
            .map(|(position, value)| self.text(&location(position), value))
            .collect::<Result<Vec<_>, KytoError>>()?;
        self.dataset(member, &texts, Some(("string-array", "0.2.0")))
    }
}
