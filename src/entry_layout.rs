use core::alloc::Layout;

use crate::constants::checked_align_up;
use crate::drop_entry::{DropEntry, value_offset_after_entry_for_align};

/// Effective alignment of the `DropEntry` slot when followed by a
/// value of alignment `value_align`.
#[inline]
#[must_use]
pub const fn entry_align_for(value_align: usize) -> usize {
    if value_align > align_of::<DropEntry>() {
        value_align
    } else {
        align_of::<DropEntry>()
    }
}

/// Layout math: given cursor `cur` and `value` layout, return
/// `(entry_offset, value_offset, end)` or `None` on overflow.
#[inline]
#[must_use]
pub fn checked_entry_value_offsets(cur: usize, value: Layout) -> Option<(usize, usize, usize)> {
    let entry_align = entry_align_for(value.align());
    let entry_addr = checked_align_up(cur, entry_align)?;
    let after_entry = entry_addr.checked_add(size_of::<DropEntry>())?;
    let value_addr = checked_align_up(after_entry, value.align())?;
    let end = value_addr.checked_add(value.size())?;
    Some((entry_addr, value_addr, end))
}

/// Unchecked layout math: same as [`checked_entry_value_offsets`] but
/// uses wrapping arithmetic, suitable when the caller has already
/// proved the request fits (typically via `try_get_chunk_for`'s worst-
/// case sizing).
///
/// # Safety
///
/// The result of the equivalent [`checked_entry_value_offsets`] call
/// must be `Some(_)`. Equivalently, `cur + worst_case_extra_with_entry(value) + value.size() + value.align() - 1`
/// must be ≤ the chunk's `total_size`.
#[inline]
#[must_use]
pub const unsafe fn entry_value_offsets_unchecked(cur: usize, value: Layout) -> (usize, usize, usize) {
    let value_align = value.align();
    let entry_align = entry_align_for(value_align);
    let entry_addr = (cur + (entry_align - 1)) & !(entry_align - 1);
    let after_entry = entry_addr + size_of::<DropEntry>();
    let value_addr = (after_entry + (value_align - 1)) & !(value_align - 1);
    let end = value_addr + value.size();
    (entry_addr, value_addr, end)
}

/// Worst-case extra bytes beyond `value.size() + value.align() - 1`
/// needed for a co-allocated `DropEntry` plus padding.
#[inline]
#[must_use]
pub const fn worst_case_extra_with_entry(value: Layout) -> usize {
    let entry_align_excess = align_of::<DropEntry>().saturating_sub(value.align());
    entry_align_excess + value_offset_after_entry_for_align(value.align())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worst_case_bounds_actual_for_all_alignments() {
        for &align in &[1_usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            for &size in &[0_usize, 1, 8, 32, 64, 4096] {
                let layout = Layout::from_size_align(size, align).unwrap();
                let baseline = size + align.saturating_sub(1);
                let extra = worst_case_extra_with_entry(layout);
                let claimed = baseline + extra;

                let entry_align = entry_align_for(align);
                let cur = entry_align - 1;
                let (_e, _v, end) = checked_entry_value_offsets(cur, layout).unwrap();
                let actual_used = end - cur;
                assert!(
                    claimed >= actual_used,
                    "worst_case underestimate: align={align} size={size}: claimed={claimed} actual={actual_used}"
                );
            }
        }
    }
}
