//! Deterministic pseudo-random number generation for the simulation core.
//!
//! The crate stays dependency-free, so this module implements xoshiro256++
//! seeded through SplitMix64 (the canonical construction from the Blackman &
//! Vina paper) plus the samplers the zero-intelligence agents need.  The whole
//! point of the simulator is determinism: identical seeds must reproduce
//! identical markets bit for bit, on every platform, forever.

/// xoshiro256++ generator seeded via SplitMix64.
#[derive(Clone, Debug)]
pub struct Rng {
    state: [u64; 4],
}

impl Rng {
    /// Builds a generator from a single 64-bit seed.
    pub fn seed_from_u64(seed: u64) -> Self {
        let mut sm = SplitMix64 { state: seed };
        Self {
            state: [sm.next(), sm.next(), sm.next(), sm.next()],
        }
    }

    /// Raw 64-bit output.
    pub fn next_u64(&mut self) -> u64 {
        let result = self.state[0]
            .wrapping_add(self.state[3])
            .rotate_left(23)
            .wrapping_add(self.state[0]);
        let t = self.state[1] << 17;
        self.state[2] ^= self.state[0];
        self.state[3] ^= self.state[1];
        self.state[1] ^= self.state[2];
        self.state[0] ^= self.state[3];
        self.state[2] ^= t;
        self.state[3] = self.state[3].rotate_left(45);
        result
    }

    /// Uniform in `[0, 1)` with 53 bits of mantissa precision.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in `[low, high]` (inclusive on both ends).
    pub fn uniform_int(&mut self, low: i64, high: i64) -> i64 {
        debug_assert!(low <= high);
        let span = (high - low + 1) as u64;
        low + (self.next_u64() % span) as i64
    }

    /// Returns `true` with probability `p`.
    pub fn bernoulli(&mut self, p: f64) -> bool {
        debug_assert!((0.0..=1.0).contains(&p));
        self.next_f64() < p
    }

    /// Unit-rate exponential draw via inverse transform.
    pub fn exp_unit(&mut self) -> f64 {
        let u = self.next_f64();
        -(1.0 - u).ln()
    }

    /// Standard normal draw via Box-Muller.
    pub fn standard_normal(&mut self) -> f64 {
        let u1 = self.next_f64().max(f64::MIN_POSITIVE);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }

    /// Poisson-process inter-arrival time in whole milliseconds for the given
    /// per-second rate.  Always at least 1 ms so events stay strictly ordered.
    pub fn poisson_gap_ms(&mut self, rate_per_second: f64) -> u64 {
        debug_assert!(rate_per_second > 0.0);
        let seconds = self.exp_unit() / rate_per_second;
        let millis = (seconds * 1000.0).round();
        millis.max(1.0) as u64
    }
}

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_reproduces_the_same_stream() {
        let mut a = Rng::seed_from_u64(42);
        let mut b = Rng::seed_from_u64(42);
        for _ in 0..1_000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn different_seeds_diverge_immediately() {
        let mut a = Rng::seed_from_u64(1);
        let mut b = Rng::seed_from_u64(2);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    #[test]
    fn uniforms_stay_in_range_and_average_near_half() {
        let mut rng = Rng::seed_from_u64(7);
        let mut sum = 0.0;
        let draws = 100_000;
        for _ in 0..draws {
            let u = rng.next_f64();
            assert!((0.0..1.0).contains(&u));
            sum += u;
        }
        let average = sum / draws as f64;
        assert!((0.495..0.505).contains(&average), "average was {average}");
    }

    #[test]
    fn uniform_int_respects_bounds() {
        let mut rng = Rng::seed_from_u64(11);
        for _ in 0..1_000 {
            let value = rng.uniform_int(-3, 5);
            assert!((-3..=5).contains(&value));
        }
    }

    #[test]
    fn poisson_gaps_are_positive_integers() {
        let mut rng = Rng::seed_from_u64(13);
        for _ in 0..1_000 {
            assert!(rng.poisson_gap_ms(2.0) >= 1);
        }
    }
}
