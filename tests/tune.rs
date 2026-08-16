//! Throwaway parameter tuning probe - deleted before commit.
use trading_simulator::sim::{NoiseAgentParams, NoiseMarket, NoiseMarketConfig};
use trading_simulator::stats;

#[allow(clippy::too_many_arguments)]
fn probe(
    name: &str,
    n_agents: usize,
    overshoot: i64,
    behind: i64,
    aggr: f64,
    cancel: f64,
    sigma: f64,
) {
    let mut market = NoiseMarket::new(NoiseMarketConfig {
        n_agents,
        params: NoiseAgentParams {
            aggressive_overshoot_max_ticks: overshoot,
            passive_max_quote_offset_ticks: behind,
            aggressive_probability: aggr,
            cancel_probability: cancel,
            size_sigma: sigma,
            ..NoiseAgentParams::default()
        },
        seed: 2026,
        ..NoiseMarketConfig::default()
    });
    market.run_until(2_400_000);
    let bars = market.bars(1_000);
    let closes: Vec<f64> = bars.iter().map(|b| b.close as f64).collect();
    let returns = stats::log_returns(&closes);
    let acf1 = stats::autocorrelation(&returns, 1);
    let acf5 = stats::autocorrelation(&returns, 5);
    let vr = stats::variance_ratio(&returns, 10);
    let kur = stats::excess_kurtosis(&returns);
    let spreads = market.spread_samples_ticks();
    let mut s = spreads.to_vec();
    s.sort_unstable();
    let med = s[s.len() / 2];
    let p95 = s[(s.len() as f64 * 0.95) as usize];
    let vols: Vec<f64> = bars.iter().map(|b| b.volume as f64).collect();
    let cv = stats::std_dev(&vols) / stats::mean(&vols);
    let sorted_vols = {
        let mut v = vols.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    let med_vol = sorted_vols[sorted_vols.len() / 2];
    let max_vol = sorted_vols[sorted_vols.len() - 1];
    let prints: Vec<f64> = market.tape().iter().map(|p| p.quantity as f64).collect();
    let print_cv = stats::std_dev(&prints) / stats::mean(&prints);
    let lo = bars.iter().map(|b| b.low).min().unwrap();
    let hi = bars.iter().map(|b| b.high).max().unwrap();
    eprintln!(
        "{name:24} acf1={acf1:+.3} acf5={acf5:+.3} vr={vr:.2} kur={kur:+6.1} med={med} p95={p95:2} bcv={cv:.2} max/med={:.1} pcv={print_cv:.2} lo/hi={lo}/{hi} prints={}",
        max_vol / med_vol,
        market.tape().len()
    );
}

#[test]
#[ignore = "parameter tuning probe; run explicitly with: cargo test --release --test tune -- --ignored --nocapture"]
fn tune() {
    // The accepted calibration and its nearest neighbours, kept as the
    // record of how tests/stylized_facts.rs parameters were chosen.
    probe("ACCEPTED n64 c45 s2.4", 64, 5, 1, 0.30, 0.45, 2.4);
    probe("n96 c45 s2.4", 96, 5, 1, 0.30, 0.45, 2.4);
    probe("n192 c45 s2.4", 192, 5, 1, 0.30, 0.45, 2.4);
    probe("n64 c35 s2.4", 64, 5, 1, 0.30, 0.35, 2.4);
    probe("n64 c50 s2.4", 64, 5, 1, 0.30, 0.50, 2.4);
}
