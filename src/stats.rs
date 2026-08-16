//! Descriptive statistics used by the stylized-facts acceptance tests.
//!
//! These are deliberately small, dependency-free implementations of exactly
//! the estimators the M1/M2 verification harness needs - nothing more.  They
//! operate on `f64` slices and panic on empty input where noted.

/// Arithmetic mean.  Panics on an empty slice.
pub fn mean(samples: &[f64]) -> f64 {
    assert!(!samples.is_empty(), "mean of empty slice");
    samples.iter().sum::<f64>() / samples.len() as f64
}

/// Population variance (second central moment).
pub fn variance(samples: &[f64]) -> f64 {
    let m = mean(samples);
    central_moment(samples, m, 2)
}

/// Sample standard deviation.
pub fn std_dev(samples: &[f64]) -> f64 {
    variance(samples).sqrt()
}

/// Excess kurtosis `g2 = m4 / m2^2 - 3`; Gaussian samples give ~0, fat tails
/// give strongly positive values.
pub fn excess_kurtosis(samples: &[f64]) -> f64 {
    let m = mean(samples);
    let m2 = central_moment(samples, m, 2);
    let m4 = central_moment(samples, m, 4);
    assert!(m2 > 0.0, "kurtosis of a constant series is undefined");
    m4 / (m2 * m2) - 3.0
}

/// Lag-`k` autocorrelation (Pearson correlation of the series against itself
/// shifted by `k`).  Returns 0 when the lag leaves too few points.
pub fn autocorrelation(samples: &[f64], lag: usize) -> f64 {
    if lag == 0 || samples.len() <= lag + 2 {
        return 0.0;
    }
    let head = &samples[..samples.len() - lag];
    let tail = &samples[lag..];
    correlation(head, tail)
}

/// Pearson correlation of two equal-length series.
pub fn correlation(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "correlation needs equal lengths");
    assert!(x.len() >= 2, "correlation needs at least two points");
    let mx = mean(x);
    let my = mean(y);
    let mut cov = 0.0;
    let mut vx = 0.0;
    let mut vy = 0.0;
    for (&xi, &yi) in x.iter().zip(y) {
        let dx = xi - mx;
        let dy = yi - my;
        cov += dx * dy;
        vx += dx * dx;
        vy += dy * dy;
    }
    if vx == 0.0 || vy == 0.0 {
        return 0.0;
    }
    cov / (vx.sqrt() * vy.sqrt())
}

/// Natural-log returns of a price series.  The output has one fewer element.
pub fn log_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|pair| (pair[1] / pair[0]).ln())
        .collect()
}

/// Variance ratio `Var(r_k) / (k * Var(r_1))`; a pure random walk gives 1.0.
pub fn variance_ratio(returns: &[f64], k: usize) -> f64 {
    assert!(k >= 2, "variance ratio needs k >= 2");
    let base = variance(returns);
    assert!(base > 0.0, "variance ratio of a constant series is undefined");
    let aggregated: Vec<f64> = returns
        .chunks(k)
        .filter(|chunk| chunk.len() == k)
        .map(|chunk| chunk.iter().sum::<f64>())
        .collect();
    variance(&aggregated) / (k as f64 * base)
}

fn central_moment(samples: &[f64], m: f64, power: u32) -> f64 {
    samples
        .iter()
        .map(|&x| {
            let d = x - m;
            d.powi(power as i32)
        })
        .sum::<f64>()
        / samples.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_variance_match_hand_computed_values() {
        let samples = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(mean(&samples), 2.5);
        assert_eq!(variance(&samples), 1.25);
    }

    #[test]
    fn excess_kurtosis_of_a_uniform_grid_is_negative() {
        // For 1..=4: m2 = 1.25, m4 = 2.5625, so g2 = 2.5625/1.5625 - 3 = -1.36.
        let samples = [1.0, 2.0, 3.0, 4.0];
        assert!((excess_kurtosis(&samples) + 1.36).abs() < 1e-12);
    }

    #[test]
    fn a_single_large_outlier_drives_kurtosis_up() {
        let mut spiky = vec![0.0; 100];
        spiky[0] = 50.0;
        assert!(excess_kurtosis(&spiky) > 10.0);
    }

    #[test]
    fn autocorrelation_of_a_linear_ramp_is_one() {
        let ramp: Vec<f64> = (0..50).map(|i| i as f64).collect();
        assert!((autocorrelation(&ramp, 1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn variance_ratio_of_a_random_walk_is_near_one() {
        // Deterministic pseudo-random walk from our own RNG.
        let mut rng = crate::rng::Rng::seed_from_u64(99);
        let mut price = 100.0;
        let mut prices = vec![price];
        for _ in 0..20_000 {
            price *= 1.0 + 0.001 * rng.standard_normal();
            prices.push(price);
        }
        let returns = log_returns(&prices);
        let ratio = variance_ratio(&returns, 10);
        assert!((0.85..1.15).contains(&ratio), "variance ratio was {ratio}");
    }
}
