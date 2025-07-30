use std::io::Write;

use shared::{Benchmark, Sample};

fn main() -> anyhow::Result<()> {
    let mut args = vec![];
    for osarg in std::env::args_os().skip(1) {
        let Ok(arg) = osarg.into_string() else {
            anyhow::bail!("all arguments must be valid UTF-8")
        };
        args.push(arg);
    }
    anyhow::ensure!(
        !args.is_empty(),
        "Usage: runner [--quiet] (<engine-name> | --version)"
    );
    if args.iter().any(|a| a == "--version") {
        writeln!(std::io::stdout(), env!("CARGO_PKG_VERSION"))?;
        return Ok(());
    }
    let quiet = args.iter().any(|a| a == "--quiet");
    let aligned = args.iter().any(|a| a == "--aligned");
    let engine = &**args.last().unwrap();
    let b = Benchmark::from_stdin()?;
    let samples = match (&*engine, &*b.model, aligned) {
        ("memchr-prebuilt", "count-bytes", false) => {
            bufchr_prebuilt_count_unaligned(&b)?
        }
        ("memchr-prebuilt", "count-bytes", true) => {
            bufchr_prebuilt_count_aligned(&b)?
        }
        ("memchr2-prebuilt", "count-bytes", false) => {
            bufchr2_prebuilt_count_unaligned(&b)?
        }
        ("memchr3-prebuilt", "count-bytes", false) => {
            bufchr3_prebuilt_count_unaligned(&b)?
        }
        (engine, model, _) => {
            anyhow::bail!("unrecognized engine '{engine}' and model '{model}'")
        }
    };
    if !quiet {
        let mut stdout = std::io::stdout().lock();
        for s in samples.iter() {
            writeln!(stdout, "{},{}", s.duration.as_nanos(), s.count)?;
        }
    }
    Ok(())
}

fn bufchr_prebuilt_count_unaligned(
    b: &Benchmark,
) -> anyhow::Result<Vec<Sample>> {
    let haystack = &b.haystack;
    let needle = b.one_needle_byte()?;
    shared::run(b, || Ok(bufchr_avx2_unaligned_iter(needle, haystack)))
}

fn bufchr_prebuilt_count_aligned(
    b: &Benchmark,
) -> anyhow::Result<Vec<Sample>> {
    let haystack = &b.haystack;
    let needle = b.one_needle_byte()?;
    shared::run(b, || Ok(bufchr_avx2_aligned_iter(needle, haystack)))
}

fn bufchr2_prebuilt_count_unaligned(
    b: &Benchmark,
) -> anyhow::Result<Vec<Sample>> {
    let haystack = &b.haystack;
    let (n1, n2) = b.two_needle_bytes()?;
    shared::run(b, || Ok(bufchr2_avx2_unaligned_iter(n1, n2, haystack)))
}

fn bufchr3_prebuilt_count_unaligned(
    b: &Benchmark,
) -> anyhow::Result<Vec<Sample>> {
    let haystack = &b.haystack;
    let (n1, n2, n3) = b.three_needle_bytes()?;
    shared::run(b, || Ok(bufchr3_avx2_unaligned_iter(n1, n2, n3, haystack)))
}

/// A trait for adding some helper routines to pointers.
pub(crate) trait Pointer {
    /// Returns the distance, in units of `T`, between `self` and `origin`.
    ///
    /// # Safety
    ///
    /// Same as `ptr::offset_from` in addition to `self >= origin`.
    unsafe fn distance(self, origin: Self) -> usize;

    /// Casts this pointer to `usize`.
    ///
    /// Callers should not convert the `usize` back to a pointer if at all
    /// possible. (And if you believe it's necessary, open an issue to discuss
    /// why. Otherwise, it has the potential to violate pointer provenance.)
    /// The purpose of this function is just to be able to do arithmetic, i.e.,
    /// computing offsets or alignments.
    fn as_usize(self) -> usize;
}

impl<T> Pointer for *const T {
    unsafe fn distance(self, origin: *const T) -> usize {
        // TODO: Replace with `ptr::sub_ptr` once stabilized.
        usize::try_from(self.offset_from(origin)).unwrap_unchecked()
    }

    fn as_usize(self) -> usize {
        self as usize
    }
}

use core::arch::x86_64::{
    __m256i, _mm256_cmpeq_epi8, _mm256_load_si256, _mm256_loadu_si256,
    _mm256_movemask_epi8, _mm256_or_si256, _mm256_set1_epi8,
};

#[inline(always)]
fn get_for_offset(mask: u32) -> u32 {
    #[cfg(target_endian = "big")]
    {
        mask.swap_bytes()
    }
    #[cfg(target_endian = "little")]
    {
        mask
    }
}

#[inline(always)]
fn first_offset(mask: u32) -> usize {
    get_for_offset(mask).trailing_zeros() as usize
}

#[inline(always)]
fn clear_least_significant_bit(mask: u32) -> u32 {
    mask & (mask - 1)
}

struct OneMatchesAligned<'h> {
    start: *const u8,
    end: *const u8,
    current: *const u8,
    mask: Option<(*const u8, u32)>,
    needle: u8,
    splat: __m256i,
    haystack: core::marker::PhantomData<&'h [u8]>,
}

const BYTES: usize = 32;
const ALIGN: usize = 31;

// NOTE: could clamp the mask to avoid scalar operations at beginning and end
impl<'h> OneMatchesAligned<'h> {
    #[target_feature(enable = "avx2")]
    unsafe fn new(needle: u8, haystack: &[u8]) -> Self {
        let ptr = haystack.as_ptr();

        Self {
            start: ptr,
            end: ptr.wrapping_add(haystack.len()),
            current: ptr,
            mask: None,
            needle,
            splat: _mm256_set1_epi8(needle as i8),
            haystack: core::marker::PhantomData,
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn next(&mut self) -> Option<usize> {
        if self.start >= self.end {
            return None;
        }

        'main: loop {
            // Processing current move mask
            if let Some((from, mask)) = &mut self.mask {
                debug_assert!(*mask != 0);

                let offset = from.add(first_offset(*mask));
                let next_mask = clear_least_significant_bit(*mask);

                if next_mask != 0 {
                    *mask = next_mask;
                } else {
                    self.mask = None;
                }

                return Some(offset.distance(self.start));
            }

            // Processing first unaligned bytes
            while self.current
                < self.start.add(BYTES - (self.start.as_usize()) & ALIGN)
            {
                if *self.current == self.needle {
                    let offset = self.current.distance(self.start);
                    self.current = self.current.add(1);
                    return Some(offset);
                } else {
                    self.current = self.current.add(1);
                }
            }

            // Main loop of aligned loads
            while self.current <= self.end.sub(BYTES) {
                debug_assert_eq!(0, self.current.as_usize() % BYTES);

                let chunk = _mm256_load_si256(self.current as *const __m256i);
                let cmp = _mm256_cmpeq_epi8(chunk, self.splat);
                let mask = _mm256_movemask_epi8(cmp) as u32;

                let next = self.current.add(BYTES);

                if mask != 0 {
                    self.mask = Some((self.current, mask));
                    self.current = next;
                    continue 'main;
                } else {
                    self.current = next;
                }
            }

            // debug_assert!(self.end.distance(self.current) < BYTES);

            // Processing remaining bytes linearly
            while self.current < self.end {
                if *self.current == self.needle {
                    let offset = self.current.distance(self.start);
                    self.current = self.current.add(1);
                    return Some(offset);
                } else {
                    self.current = self.current.add(1);
                }
            }

            return None;
        }
    }
}

struct OneMatchesUnaligned<'h> {
    splat: __m256i,
    start: *const u8,
    end: *const u8,
    current: *const u8,
    mask: u32,
    needle: u8,
    haystack: core::marker::PhantomData<&'h [u8]>,
}

impl<'h> OneMatchesUnaligned<'h> {
    #[target_feature(enable = "avx2")]
    unsafe fn new(needle: u8, haystack: &[u8]) -> Self {
        // dbg!(size_of::<Self>(), align_of::<Self>());
        let ptr = haystack.as_ptr();

        Self {
            start: ptr,
            end: ptr.wrapping_add(haystack.len()),
            current: ptr,
            mask: 0,
            needle,
            splat: _mm256_set1_epi8(needle as i8),
            haystack: core::marker::PhantomData,
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn next(&mut self) -> Option<usize> {
        if self.start >= self.end {
            return None;
        }

        let mut mask = self.mask;
        let vectorized_end = self.end.sub(BYTES);
        let mut current = self.current;
        let start = self.start;
        let splat = self.splat;

        'main: loop {
            // Processing current move mask
            if mask != 0 {
                let offset = current.sub(BYTES).add(first_offset(mask));
                self.mask = clear_least_significant_bit(mask);
                self.current = current;

                return Some(offset.distance(start));
            }

            // Main loop of unaligned loads
            while current <= vectorized_end {
                let chunk = _mm256_loadu_si256(current as *const __m256i);
                let cmp = _mm256_cmpeq_epi8(chunk, splat);
                mask = _mm256_movemask_epi8(cmp) as u32;

                current = current.add(BYTES);

                if mask != 0 {
                    continue 'main;
                }
            }

            // Processing remaining bytes linearly
            while current < self.end {
                if *current == self.needle {
                    let offset = current.distance(start);
                    self.current = current.add(1);
                    return Some(offset);
                }
                current = current.add(1);
            }

            return None;
        }
    }
}

struct TwoMatchesUnaligned<'h> {
    splat1: __m256i,
    splat2: __m256i,
    start: *const u8,
    end: *const u8,
    current: *const u8,
    mask: u32,
    needle1: u8,
    needle2: u8,
    haystack: core::marker::PhantomData<&'h [u8]>,
}

impl<'h> TwoMatchesUnaligned<'h> {
    #[target_feature(enable = "avx2")]
    unsafe fn new(needle1: u8, needle2: u8, haystack: &[u8]) -> Self {
        // dbg!(size_of::<Self>(), align_of::<Self>());
        let ptr = haystack.as_ptr();

        Self {
            start: ptr,
            end: ptr.wrapping_add(haystack.len()),
            current: ptr,
            mask: 0,
            needle1,
            needle2,
            splat1: _mm256_set1_epi8(needle1 as i8),
            splat2: _mm256_set1_epi8(needle2 as i8),
            haystack: core::marker::PhantomData,
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn next(&mut self) -> Option<usize> {
        if self.start >= self.end {
            return None;
        }

        let mut mask = self.mask;
        let vectorized_end = self.end.sub(BYTES);
        let mut current = self.current;
        let start = self.start;

        'main: loop {
            // Processing current move mask
            if mask != 0 {
                let offset = current.sub(BYTES).add(first_offset(mask));
                self.mask = clear_least_significant_bit(mask);
                self.current = current;

                return Some(offset.distance(start));
            }

            // Main loop of unaligned loads
            while current <= vectorized_end {
                let chunk = _mm256_loadu_si256(current as *const __m256i);
                let cmp1 = _mm256_cmpeq_epi8(chunk, self.splat1);
                let cmp2 = _mm256_cmpeq_epi8(chunk, self.splat2);
                let cmp = _mm256_or_si256(cmp1, cmp2);

                mask = _mm256_movemask_epi8(cmp) as u32;

                current = current.add(BYTES);

                if mask != 0 {
                    continue 'main;
                }
            }

            // Processing remaining bytes linearly
            while current < self.end {
                if *current == self.needle1 || *current == self.needle2 {
                    let offset = current.distance(start);
                    self.current = current.add(1);
                    return Some(offset);
                }
                current = current.add(1);
            }

            return None;
        }
    }
}

struct ThreeMatchesUnaligned<'h> {
    splat1: __m256i,
    splat2: __m256i,
    splat3: __m256i,
    start: *const u8,
    end: *const u8,
    current: *const u8,
    mask: u32,
    needle1: u8,
    needle2: u8,
    needle3: u8,
    haystack: core::marker::PhantomData<&'h [u8]>,
}

impl<'h> ThreeMatchesUnaligned<'h> {
    #[target_feature(enable = "avx2")]
    unsafe fn new(
        needle1: u8,
        needle2: u8,
        needle3: u8,
        haystack: &[u8],
    ) -> Self {
        // dbg!(size_of::<Self>(), align_of::<Self>());
        let ptr = haystack.as_ptr();

        Self {
            start: ptr,
            end: ptr.wrapping_add(haystack.len()),
            current: ptr,
            mask: 0,
            needle1,
            needle2,
            needle3,
            splat1: _mm256_set1_epi8(needle1 as i8),
            splat2: _mm256_set1_epi8(needle2 as i8),
            splat3: _mm256_set1_epi8(needle3 as i8),
            haystack: core::marker::PhantomData,
        }
    }

    #[target_feature(enable = "avx2")]
    unsafe fn next(&mut self) -> Option<usize> {
        if self.start >= self.end {
            return None;
        }

        let mut mask = self.mask;
        let vectorized_end = self.end.sub(BYTES);
        let mut current = self.current;
        let start = self.start;

        'main: loop {
            // Processing current move mask
            if mask != 0 {
                let offset = current.sub(BYTES).add(first_offset(mask));
                self.mask = clear_least_significant_bit(mask);
                self.current = current;

                return Some(offset.distance(start));
            }

            // Main loop of unaligned loads
            while current <= vectorized_end {
                let chunk = _mm256_loadu_si256(current as *const __m256i);
                let cmp1 = _mm256_cmpeq_epi8(chunk, self.splat1);
                let cmp2 = _mm256_cmpeq_epi8(chunk, self.splat2);
                let cmp3 = _mm256_cmpeq_epi8(chunk, self.splat3);
                let cmp = _mm256_or_si256(cmp1, cmp2);
                let cmp = _mm256_or_si256(cmp, cmp3);

                mask = _mm256_movemask_epi8(cmp) as u32;

                current = current.add(BYTES);

                if mask != 0 {
                    continue 'main;
                }
            }

            // Processing remaining bytes linearly
            while current < self.end {
                if *current == self.needle1
                    || *current == self.needle2
                    || *current == self.needle3
                {
                    let offset = current.distance(start);
                    self.current = current.add(1);
                    return Some(offset);
                }
                current = current.add(1);
            }

            return None;
        }
    }
}

struct OneMatchesUnalignedIter<'h>(OneMatchesUnaligned<'h>);

impl<'h> OneMatchesUnalignedIter<'h> {
    fn new(needle: u8, haystack: &[u8]) -> Self {
        unsafe { Self(OneMatchesUnaligned::new(needle, haystack)) }
    }
}

impl<'h> Iterator for OneMatchesUnalignedIter<'h> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.0.next() }
    }
}

struct OneMatchesAlignedIter<'h>(OneMatchesAligned<'h>);

impl<'h> OneMatchesAlignedIter<'h> {
    fn new(needle: u8, haystack: &[u8]) -> Self {
        unsafe { Self(OneMatchesAligned::new(needle, haystack)) }
    }
}

impl<'h> Iterator for OneMatchesAlignedIter<'h> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.0.next() }
    }
}

struct TwoMatchesUnalignedIter<'h>(TwoMatchesUnaligned<'h>);

impl<'h> TwoMatchesUnalignedIter<'h> {
    fn new(needle1: u8, needle2: u8, haystack: &[u8]) -> Self {
        unsafe { Self(TwoMatchesUnaligned::new(needle1, needle2, haystack)) }
    }
}

impl<'h> Iterator for TwoMatchesUnalignedIter<'h> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.0.next() }
    }
}

struct ThreeMatchesUnalignedIter<'h>(ThreeMatchesUnaligned<'h>);

impl<'h> ThreeMatchesUnalignedIter<'h> {
    fn new(needle1: u8, needle2: u8, needle3: u8, haystack: &[u8]) -> Self {
        unsafe { Self(ThreeMatchesUnaligned::new(needle1, needle2, needle3, haystack)) }
    }
}

impl<'h> Iterator for ThreeMatchesUnalignedIter<'h> {
    type Item = usize;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        unsafe { self.0.next() }
    }
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn bufchr_avx2_unaligned_iter(needle: u8, haystack: &[u8]) -> usize {
    OneMatchesUnalignedIter::new(needle, haystack).count()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn bufchr_avx2_aligned_iter(needle: u8, haystack: &[u8]) -> usize {
    OneMatchesAlignedIter::new(needle, haystack).count()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn bufchr2_avx2_unaligned_iter(
    needle1: u8,
    needle2: u8,
    haystack: &[u8],
) -> usize {
    TwoMatchesUnalignedIter::new(needle1, needle2, haystack).count()
}

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub fn bufchr3_avx2_unaligned_iter(
    needle1: u8,
    needle2: u8,
    needle3: u8,
    haystack: &[u8],
) -> usize {
    ThreeMatchesUnalignedIter::new(needle1, needle2, needle3, haystack).count()
}
