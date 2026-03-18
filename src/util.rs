/// Integer hash → [0, 1).
pub fn hash01(n: u32) -> f32 {
    let n = n.wrapping_mul(747796405).wrapping_add(2891336453);
    let n = ((n >> ((n >> 28).wrapping_add(4))) ^ n).wrapping_mul(277803737);
    let n = (n >> 22) ^ n;
    (n as f32) / (u32::MAX as f32)
}

/// Hermite smoothstep.
pub fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Splitmix64 PRNG — deterministic, no external dependency.
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }

    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    pub fn next_f32_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}
