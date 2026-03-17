/// Reserved visibility-buffer value for empty pixels.
pub const VISIBILITY_EMPTY: u32 = 0;
/// Bias applied so valid geometry never aliases the empty sentinel.
pub const VISIBILITY_ID_BIAS: u32 = 1;
pub fn pack_visibility_id(meshlet_idx: u32, tri_idx: u32) -> u32 {
    ((meshlet_idx << 8) | tri_idx) + VISIBILITY_ID_BIAS
}

pub fn unpack_visibility_id(vis_id: u32) -> Option<(u32, u32)> {
    if vis_id == VISIBILITY_EMPTY {
        return None;
    }
    let packed = vis_id - VISIBILITY_ID_BIAS;
    Some((packed >> 8, packed & 0xFF))
}

#[cfg(test)]
mod tests {
    use super::{pack_visibility_id, unpack_visibility_id, VISIBILITY_EMPTY};

    #[test]
    fn visibility_id_roundtrips() {
        let packed = pack_visibility_id(17, 23);
        assert_ne!(packed, VISIBILITY_EMPTY);
        assert_eq!(unpack_visibility_id(packed), Some((17, 23)));
        assert_eq!(unpack_visibility_id(VISIBILITY_EMPTY), None);
    }
}
