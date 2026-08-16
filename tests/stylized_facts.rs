//! M1 acceptance harness: stylized facts from the zero-intelligence market.
//!
//! The core claim under test is the Farmer et al. zero-intelligence result:
//! matching-engine mechanics plus uninformed random order flow already
//! reproduce the salient statistical regularities of real order-driven
//! markets - tight positive spreads, negligible return autocorrelation,
//! near-random-walk diffusion, and heavy-tailed volume - before any
//! behavioural sophistication (M2) is added.
//!
//! The market parameters and the seed are fixed, so every assertion is
//! reproducible bit for bit.  Diagnostics print with `--nocapture`.

use trading_simulator::sim::{NoiseMarket, NoiseMarketConfig, NoiseAgentParams};
use trading_simulator::stats;

/// 192 agents waking ~once per second for 40 simulated minutes.
fn acceptance_config(seed: u64) -> NoiseMarketConfig {
    NoiseMarketConfig {
        ref_price: 1_000,
        n_agents: 192,
        day_length_ms: 1_200_000,
        params: NoiseAgentParams {
            wake_rate_per_second: 1.0,
            cancel_probability: 0.35,
            aggressive_probability: 0.30,
            size_median_lots: 1.0,
            size_sigma: 1.0,
        },
        seed,
        ..NoiseMarketConfig::default()
    }
}

fn percentile(sorted: &[i64], p: f64) -> i64 {
    let index = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[test]
fn zero_intelligence_market_reproduces_stylized_facts() {
    let mut market = NoiseMarket::new(acceptance_config(2026));
    market.run_until(2_400_000);

    let tape = market.tape();
    assert!(tape.len() > 30_000, "market was too quiet: {} prints", tape.len());

    // --- Spreads: positive, tight, with a bounded tail -------------------
    let spreads = market.spread_samples_ticks();
    assert!(!spreads.is_empty());
    assert!(
        spreads.iter().all(|&s| s >= 1),
        "spreads must stay positive"
    );
    let mut sorted = spreads.to_vec();
    sorted.sort_unstable();
    let median_spread = percentile(&sorted, 0.5);
    let p95_spread = percentile(&sorted, 0.95);
    assert!(
        (1..=6).contains(&median_spread),
        "median spread {median_spread} ticks is unrealistic"
    );
    assert!(
        p95_spread <= 40,
        "spread tail too wide: p95 = {p95_spread} ticks"
    );

    // --- Bars and returns -------------------------------------------------
    let bars = market.bars(1_000);
    assert!(bars.len() >= 1_500, "only {} bars produced", bars.len());
    let closes: Vec<f64> = bars.iter().map(|bar| bar.close as f64).collect();
    let returns = stats::log_returns(&closes);

    // Weak-form efficiency: negligible linear predictability.
    let acf1 = stats::autocorrelation(&returns, 1);
    let acf5 = stats::autocorrelation(&returns, 5);
    assert!(
        acf1.abs() < 0.15 && acf5.abs() < 0.10,
        "returns look predictable: acf(1)={acf1:.3}, acf(5)={acf5:.3}"
    );

    // Diffusion close to a random walk (slight mean reversion is expected
    // from mid-anchored quoting, hence the generous band).
    let vr10 = stats::variance_ratio(&returns, 10);
    assert!(
        (0.55..=1.70).contains(&vr10),
        "diffusion scaling off: VR(10) = {vr10:.3}"
    );

    // Prices stay in a sane band around the reference.
    let lo = bars.iter().map(|b| b.low).min().unwrap();
    let hi = bars.iter().map(|b| b.high).max().unwrap();
    assert!(lo > 0);
    assert!(
        (500..=2_000).contains(&lo) && (500..=2_000).contains(&hi),
        "price wandered too far: [{lo}, {hi}]"
    );

    // --- Volume: heavy tailed ---------------------------------------------
    let volumes: Vec<f64> = bars.iter().map(|bar| bar.volume as f64).collect();
    let mean_volume = stats::mean(&volumes);
    let cv = stats::std_dev(&volumes) / mean_volume;
    assert!(cv > 0.6, "bar volume too uniform: CV = {cv:.3}");
    let mut sorted_volumes = volumes.clone();
    sorted_volumes.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median_volume = sorted_volumes[sorted_volumes.len() / 2];
    let max_volume = sorted_volumes[sorted_volumes.len() - 1];
    assert!(
        max_volume > 4.0 * median_volume,
        "volume tail too thin: max/median = {:.1}",
        max_volume / median_volume
    );

    // --- Diagnostics (visible with --nocapture) ----------------------------
    eprintln!("prints: {}", tape.len());
    eprintln!("bars(1s): {}", bars.len());
    eprintln!("spread median/p95: {median_spread}/{p95_spread} ticks");
    eprintln!(
        "return acf(1)/acf(5): {acf1:.4}/{acf5:.4}, VR(10): {vr10:.3}"
    );
    eprintln!(
        "excess kurtosis(1s): {:.2}",
        stats::excess_kurtosis(&returns)
    );
    eprintln!("volume CV: {cv:.2}, max/median: {:.1}", max_volume / median_volume);
}
