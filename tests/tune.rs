//! Throwaway parameter tuning probe - deleted before commit.
use trading_simulator::sim::{NoiseAgentParams, NoiseMarket, NoiseMarketConfig};
use trading_simulator::stats;

fn probe(name: &str, n_agents: usize, overshoot: i64, aggr: f64, cancel: f64, sigma: f64) {
    let mut market = NoiseMarket::new(NoiseMarketConfig {
        n_agents,
        params: NoiseAgentParams {
            aggressive_overshoot_max_ticks: overshoot,
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
    let vr = stats::variance_ratio(&returns, 10);
    let kur = stats::excess_kurtosis(&returns);
    let spreads = market.spread_samples_ticks();
    let mut s = spreads.to_vec();
    s.sort_unstable();
    let med = s[s.len() / 2];
    let p95 = s[(s.len() as f64 * 0.95) as usize];
    let vols: Vec<f64> = bars.iter().map(|b| b.volume as f64).collect();
    let cv = stats::std_dev(&vols) / stats::mean(&vols);
    let prints: Vec<f64> = market.tape().iter().map(|p| p.quantity as f64).collect();
    let print_cv = stats::std_dev(&prints) / stats::mean(&prints);
    eprintln!(
        "{name:22} acf1={acf1:+.3} vr={vr:.2} kur={kur:+6.1} med={med} p95={p95:2} bcv={cv:.2} pcv={print_cv:.2} rej={} prints={}",
        market.rejected_submits(),
        market.tape().len()
    );
}

#[test]
fn tune() {
    probe("A o2 a30 c35 s1.0", 192, 2, 0.30, 0.35, 1.0);
    probe("B o2 a20 c35 s1.0", 192, 2, 0.20, 0.35, 1.0);
    probe("C o5 a20 c35 s1.0", 192, 5, 0.20, 0.35, 1.0);
    probe("D o5 a20 c50 s1.0", 192, 5, 0.20, 0.50, 1.0);
    probe("E o5 a15 c35 s1.4", 192, 5, 0.15, 0.35, 1.4);
    probe("F 256 o5 a20 c35", 256, 5, 0.20, 0.35, 1.0);
}
