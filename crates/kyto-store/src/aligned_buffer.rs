//! A growable buffer whose storage starts on a 64-byte boundary (plan 5.1).
//!
//! Does: hold plain values (`u64`, `u32`, `f32`) in 64-byte lines, so the first value
//! is 64-byte aligned (a cache line, an AVX-512 register) and SIMD kernels (task 5.3)
//! can use aligned loads; grow like a `Vec` (capacity doubles), keep its capacity when
//! cleared, and change its element type without reallocating, so a pool (kyto-exec
//! `CellBlockPool`) can reuse one allocation for `float32` and `uint32` values.
//! Does not: pool itself (kyto-exec does), or use `unsafe`: the lines are cast to
//! values through `bytemuck`.

use std::fmt;
use std::marker::PhantomData;

use bytemuck::{Pod, Zeroable};

/// The alignment, in bytes, of every [`AlignedBuffer`]'s first value.
pub const BUFFER_ALIGNMENT: usize = 64;

/// One 64-byte, 64-byte aligned line of storage.
#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C, align(64))]
struct Line([u8; BUFFER_ALIGNMENT]);

/// A growable buffer of plain values whose first value is 64-byte aligned.
///
/// Reads and writes go through slices ([`AlignedBuffer::as_slice`],
/// [`AlignedBuffer::as_mut_slice`]). Capacity grows by doubling, as `Vec`'s does, and
/// is kept by [`AlignedBuffer::clear`] and [`AlignedBuffer::into_cleared`]; spare
/// capacity is whole lines of zeros.
pub struct AlignedBuffer<T> {
    lines: Vec<Line>,
    len: usize,
    element: PhantomData<T>,
}

impl<T: Pod> AlignedBuffer<T> {
    /// Values per line. Fails to compile for a type whose size does not divide 64 or
    /// whose alignment is above 64.
    const PER_LINE: usize = {
        assert!(size_of::<T>() > 0 && BUFFER_ALIGNMENT.is_multiple_of(size_of::<T>()));
        assert!(align_of::<T>() <= BUFFER_ALIGNMENT);
        BUFFER_ALIGNMENT / size_of::<T>()
    };

    /// An empty buffer; allocates nothing.
    pub const fn new() -> Self {
        Self {
            lines: Vec::new(),
            len: 0,
            element: PhantomData,
        }
    }

    /// An empty buffer with room for at least `capacity` values.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            lines: Vec::with_capacity(capacity.div_ceil(Self::PER_LINE)),
            len: 0,
            element: PhantomData,
        }
    }

    /// Number of values.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True when there are no values.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Number of values the buffer holds without allocating.
    pub fn capacity(&self) -> usize {
        self.lines.capacity() * Self::PER_LINE
    }

    /// The values.
    pub fn as_slice(&self) -> &[T] {
        &bytemuck::cast_slice(&self.lines)[..self.len]
    }

    /// The values, writable.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut bytemuck::cast_slice_mut(&mut self.lines)[..self.len]
    }

    /// Append `value`.
    pub fn push(&mut self, value: T) {
        if self.len == self.lines.len() * Self::PER_LINE {
            // Vec's amortised doubling, one line at a time.
            self.lines.push(Line::zeroed());
        }
        bytemuck::cast_slice_mut(&mut self.lines)[self.len] = value;
        self.len += 1;
    }

    /// Append `values`.
    pub fn extend_from_slice(&mut self, values: &[T]) {
        let start = self.len;
        let end = start + values.len();
        self.grow_lines(end);
        bytemuck::cast_slice_mut(&mut self.lines)[start..end].copy_from_slice(values);
        self.len = end;
    }

    /// Make room for at least `additional` more values (as `Vec::reserve`).
    pub fn reserve(&mut self, additional: usize) {
        let needed = (self.len + additional).div_ceil(Self::PER_LINE);
        self.lines.reserve(needed.saturating_sub(self.lines.len()));
    }

    /// Remove every value; the capacity stays.
    pub fn clear(&mut self) {
        self.len = 0;
        self.lines.clear();
    }

    /// The same allocation as an empty buffer of another value type (no copy, no
    /// allocation): a pool reuses one buffer for `float32` and `uint32` values.
    pub fn into_cleared<U: Pod>(mut self) -> AlignedBuffer<U> {
        self.lines.clear();
        AlignedBuffer {
            lines: self.lines,
            len: 0,
            element: PhantomData,
        }
    }

    /// Hold enough zeroed lines for `len` values.
    fn grow_lines(&mut self, len: usize) {
        let lines = len.div_ceil(Self::PER_LINE);
        if lines > self.lines.len() {
            self.lines.resize(lines, Line::zeroed());
        }
    }
}

impl<T: Pod> Default for AlignedBuffer<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Pod> From<&[T]> for AlignedBuffer<T> {
    /// A copy of `values` (one allocation of exactly the lines they need).
    fn from(values: &[T]) -> Self {
        let mut buffer = Self::with_capacity(values.len());
        buffer.extend_from_slice(values);
        buffer
    }
}

impl<T: Pod> Clone for AlignedBuffer<T> {
    /// A copy with exactly the lines the values need.
    fn clone(&self) -> Self {
        Self::from(self.as_slice())
    }
}

impl<T: Pod + PartialEq> PartialEq for AlignedBuffer<T> {
    /// Equal values; capacity does not matter.
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<T: Pod + fmt::Debug> fmt::Debug for AlignedBuffer<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.as_slice()).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::{AlignedBuffer, BUFFER_ALIGNMENT};

    fn aligned<T: bytemuck::Pod>(buffer: &AlignedBuffer<T>) -> bool {
        (buffer.as_slice().as_ptr() as usize).is_multiple_of(BUFFER_ALIGNMENT)
    }

    #[test]
    fn values_start_on_a_64_byte_boundary() {
        let mut buffer = AlignedBuffer::<u32>::new();
        assert!(aligned(&buffer));
        for value in 0..1000 {
            buffer.push(value);
            assert!(aligned(&buffer));
        }
        assert_eq!(buffer.as_slice(), (0..1000).collect::<Vec<_>>());
        let wide = AlignedBuffer::<u64>::from(&[1_u64, 2, 3][..]);
        assert!(aligned(&wide));
        assert_eq!(wide.as_slice(), [1, 2, 3]);
    }

    #[test]
    fn capacity_survives_clear_and_retype() {
        let mut buffer = AlignedBuffer::<u32>::new();
        buffer.extend_from_slice(&[7; 100]);
        assert_eq!(buffer.len(), 100);
        let capacity = buffer.capacity();
        assert!(capacity >= 100);
        buffer.clear();
        assert!(buffer.is_empty());
        assert_eq!(buffer.capacity(), capacity);
        let mut floats: AlignedBuffer<f32> = buffer.into_cleared();
        assert!(floats.is_empty());
        assert_eq!(floats.capacity(), capacity);
        floats.push(1.5);
        assert_eq!(floats.as_slice(), [1.5]);
    }

    #[test]
    fn grows_like_a_vec_and_reserves() {
        let mut buffer = AlignedBuffer::<f32>::with_capacity(17);
        assert_eq!(buffer.capacity(), 32);
        buffer.reserve(100);
        assert!(buffer.capacity() >= 100);
        buffer.extend_from_slice(&[1.0; 3]);
        buffer.as_mut_slice()[1] = 2.0;
        assert_eq!(buffer.as_slice(), [1.0, 2.0, 1.0]);
        assert_eq!(buffer.clone(), buffer);
        assert_eq!(format!("{buffer:?}"), "[1.0, 2.0, 1.0]");
    }
}
