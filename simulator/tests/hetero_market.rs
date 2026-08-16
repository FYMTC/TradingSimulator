//! M3 acceptance harness: heterogeneous strategies + the scenario engine.
//!
//! Three claims under test, on top of the M1/M2 microstructure
//! invariants:
//!
//! 1. The mixed population keeps the book healthy: spreads stay
//!    positive, the book never locks or crosses, every strategy
//!    population actually trades, and no submission trips a risk check.
//! 2. The scenario engine generates the *new* stylized facts the
//!    zero-intelligence market cannot: volatility clustering (|r|
//!    autocorrelation that is positive and decays), the volume-
//!    volatility relation, and a U-shaped intraday volume profile.
//! 3. Returns stay close to weak-form efficiency despite the trend
//!    crowd: linear autocorrelation of returns remains small.
//!
//! Parameters and seeds are fixed, so every assertion is reproducible
//! bit for bit.  Diagnostics print with `--nocapture`.

use trading_simulator::hetero::{HeteroMarket, HeteroMarketConfig};
use trading_simulator::stats;

/// 32 noise agents for the liquidity base, strategy populations on top,
/// six 20-minute trading days.
fn acceptance_config(seed: u64) -> HeteroMarketConfig {
    HeteroMarketConfig {
        n_noise: 32,
        n_market_makers: 3,
        n_trend: 6,
        n_mean_revert: 6,
        n_fundamental: 3,
        seed,
        ..HeteroMarketConfig::default()
    }
}

const RUN_MS: u64 = 7_200_000;

#[test]
fn mixed_population_keeps_book_invariants() {
    let mut market = HeteroMarket::new(acceptance_config(2026));
    market.run_until(RUN_MS);

    // Spreads positive whenever sampled; book never locks or crosses.
    let spreads = market.spread_samples_ticks();
    assert!(!spreads.is_empty());
    assert!(
        spreads.iter().all(|&s| s >= 1),
        "spreads must stay positive"
    );
    if let (Some(bid), Some(ask)) = (
        market.exchange().book().best_bid(),
        market.exchange().book().best_ask(),
    ) {
        assert!(bid < ask, "book must never lock or cross");
    }

    // Liquid market, internally consistent bars, no risk-check rejects.
    assert!(market.tape().len() > 20_000, "market too quiet");
    assert_eq!(market.rejected_submits(), 0);
    let bars = market.bars(1_000);
    assert!(bars.len() > 3_000, "only {} bars produced", bars.len());
    for bar in &bars {
        assert!(bar.high >= bar.low);
        assert!(bar.high >= bar.open && bar.high >= bar.close);
        assert!(bar.low <= bar.open && bar.low <= bar.close);
    }

    // Every strategy population actually participated in the tape.
    let cfg = acceptance_config(2026);
    let first_mm = cfg.n_noise as u64;
    let first_trend = first_mm + cfg.n_market_makers as u64;
    let first_mean_rev = first_trend + cfg.n_trend as u64;
    let first_fund = first_mean_rev + cfg.n_mean_revert as u64;
    let trades = market.exchange().trades();
    let involved = |lo: u64, hi: u64| {
        trades.iter().any(|trade| {
            (trade.buyer_account_id >= lo && trade.buyer_account_id < hi)
                || (trade.seller_account_id >= lo && trade.seller_account_id < hi)
        })
    };
    assert!(involved(first_mm, first_trend), "market makers traded");
    assert!(involved(first_trend, first_mean_rev), "trend agents traded");
    assert!(
        involved(first_mean_rev, first_fund),
        "mean reverters traded"
    );
    assert!(involved(first_fund, u64::MAX), "fundamentalists traded");
}

#[test]
fn scenario_engine_generates_volatility_clustering() {
    for seed in [2026u64, 33] {
        let mut market = HeteroMarket::new(acceptance_config(seed));
        market.run_until(RUN_MS);

        let bars = market.bars(1_000);
        let closes: Vec<f64> = bars.iter().map(|bar| bar.close as f64).collect();
        let returns = stats::log_returns(&closes);
        let abs_returns: Vec<f64> = returns.iter().map(|r| r.abs()).collect();
        let volumes: Vec<f64> = bars.iter().map(|bar| bar.volume as f64).collect();

        // The regime chain visited at least three regimes.
        let distinct = market
            .regime_history()
            .iter()
            .map(|(_, regime)| *regime)
            .collect::<std::collections::BTreeSet<_>>();
        assert!(
            distinct.len() >= 3,
            "the chain should visit several regimes, got {distinct:?}"
        );

        // Volatility clustering: |r| autocorrelation is clearly positive
        // at lag 1 and decays with the lag (GARCH-style memory).
        let abs_acf1 = stats::autocorrelation(&abs_returns, 1);
        let abs_acf5 = stats::autocorrelation(&abs_returns, 5);
        let abs_acf20 = stats::autocorrelation(&abs_returns, 20);
        assert!(
            abs_acf1 > 0.10,
            "no volatility clustering: |r| acf(1) = {abs_acf1:.3} (seed {seed})"
        );
        assert!(
            abs_acf5 < abs_acf1 && abs_acf20 < abs_acf5,
            "clustering should decay: acf(1)/5/20 = {abs_acf1:.3}/{abs_acf5:.3}/{abs_acf20:.3} (seed {seed})"
        );

        // Volume-volatility relation: busy bars are also violent bars.
        let corr = stats::correlation(&abs_returns, &volumes[1..]);
        assert!(
            corr > 0.10,
            "volume and volatility should correlate, got {corr:.3} (seed {seed})"
        );

        // Returns stay close to weak-form efficiency despite the trend
        // crowd.
        let acf1 = stats::autocorrelation(&returns, 1);
        assert!(
            acf1.abs() < 0.25,
            "returns look too predictable: acf(1) = {acf1:.3} (seed {seed})"
        );

        eprintln!(
            "seed {seed}: |r| acf(1)/5/20 = {abs_acf1:.3}/{abs_acf5:.3}/{abs_acf20:.3}, corr(|r|,vol) = {corr:.3}, acf(1) = {acf1:+.3}, regimes = {distinct:?}"
        );
    }
}

#[test]
fn intraday_volume_follows_a_u_shape() {
    let mut market = HeteroMarket::new(acceptance_config(2026));
    market.run_until(RUN_MS);

    // Bucket bar volumes by intraday position: session edges (first and
    // last 15%) vs the midday trough (35%..65%).
    let day_length = market
        .bars(1_000)
        .first()
        .map(|_| acceptance_config(2026).day_length_ms)
        .unwrap();
    let mut edge_vols = Vec::new();
    let mut mid_vols = Vec::new();
    for bar in market.bars(1_000) {
        let f = (bar.start_ms % day_length) as f64 / day_length as f64;
        if !(0.15..=0.85).contains(&f) {
            edge_vols.push(bar.volume as f64);
        } else if (0.35..0.65).contains(&f) {
            mid_vols.push(bar.volume as f64);
        }
    }
    assert!(!edge_vols.is_empty() && !mid_vols.is_empty());
    let ratio = stats::mean(&edge_vols) / stats::mean(&mid_vols);
    eprintln!("intraday edge/midday mean volume ratio: {ratio:.2}");
    assert!(
        ratio > 1.5,
        "session edges should trade more than midday, ratio = {ratio:.2}"
    );
}
