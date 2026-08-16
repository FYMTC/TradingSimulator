//! M2 acceptance harness: a mixed noise + reinforcement-learning market.
//!
//! Two claims under test:
//!
//! 1. Adding learning investors does not break the market microstructure
//!    the M1 harness verified - spreads stay positive and tight, the book
//!    never locks or crosses, and bars stay internally consistent.
//! 2. Learning is real: agents trained with Q-learning end up materially
//!    richer than agents with frozen (non-learning, perpetually exploring)
//!    policies running in the identical market, across several seeds.

use trading_simulator::rl::{RlAgentParams, RlMarket, RlMarketConfig};
use trading_simulator::stats;

/// 48 noise agents for background liquidity, 8 RL investors on top.
fn mixed_config(seed: u64) -> RlMarketConfig {
    RlMarketConfig {
        n_noise_agents: 48,
        n_rl_agents: 8,
        seed,
        ..RlMarketConfig::default()
    }
}

#[test]
fn mixed_market_keeps_book_and_bar_invariants() {
    let mut market = RlMarket::new(mixed_config(7));
    market.run_until(600_000);

    // Spreads: positive whenever sampled.
    let spreads = market.spread_samples_ticks();
    assert!(!spreads.is_empty());
    assert!(
        spreads.iter().all(|&s| s >= 1),
        "spreads must stay positive"
    );

    // The book must never lock or cross.
    if let (Some(bid), Some(ask)) = (
        market.exchange().book().best_bid(),
        market.exchange().book().best_ask(),
    ) {
        assert!(bid < ask, "book must never lock or cross");
    }

    // The market is liquid and the RL crowd participates.
    assert!(market.tape().len() > 1_000, "market too quiet");
    let first_rl = 48u64;
    assert!(market.exchange().trades().iter().any(|trade| {
        trade.buyer_account_id >= first_rl || trade.seller_account_id >= first_rl
    }));

    // Bars stay internally consistent.
    let bars = market.bars(1_000);
    assert!(bars.len() > 300, "only {} bars produced", bars.len());
    for bar in &bars {
        assert!(bar.high >= bar.low);
        assert!(bar.high >= bar.open && bar.high >= bar.close);
        assert!(bar.low <= bar.open && bar.low <= bar.close);
    }

    // Every RL agent actually decided (and therefore learned) all along.
    let rl = market.rl_stats();
    assert_eq!(rl.len(), 8);
    assert!(rl.iter().all(|s| s.decisions > 50));
    assert_eq!(market.rejected_submits(), 0);
}

#[test]
fn learning_agents_outperform_frozen_policies() {
    let frozen_params = RlAgentParams {
        alpha: 0.0,
        epsilon_start: 0.4,
        epsilon_min: 0.4,
        epsilon_decay: 1.0,
        ..RlAgentParams::default()
    };

    let mut learned_gains: Vec<f64> = Vec::new();
    let mut frozen_gains: Vec<f64> = Vec::new();
    let mut learned_late_rewards: Vec<f64> = Vec::new();
    for seed in [11u64, 22, 33] {
        // Trained: Q-learning with annealing exploration.
        let mut market = RlMarket::new(mixed_config(seed));
        market.run_until(900_000);
        let stats = market.rl_stats();
        learned_gains.extend(stats.iter().map(|s| s.pnl as f64 / 100.0));
        // Reward per decision in the trained second half, where the
        // annealed policy should be trading near break-even.
        for s in &stats {
            let half = s.rewards.len() / 2;
            learned_late_rewards.extend_from_slice(&s.rewards[half..]);
        }

        // Frozen: identical market, but the policy never updates and
        // exploration never anneals, so agents keep paying for random
        // spread crossing the whole run.
        let mut frozen = RlMarket::new(RlMarketConfig {
            rl: frozen_params,
            ..mixed_config(seed)
        });
        frozen.run_until(900_000);
        frozen_gains.extend(frozen.rl_stats().iter().map(|s| s.pnl as f64 / 100.0));
    }

    let learned_mean = stats::mean(&learned_gains);
    let frozen_mean = stats::mean(&frozen_gains);
    eprintln!(
        "mean trading PnL (per-share ticks): learned {learned_mean:.2} vs frozen {frozen_mean:.2}"
    );
    assert!(
        learned_mean > frozen_mean + 10.0,
        "learning should clearly beat frozen policies: {learned_mean:.2} vs {frozen_mean:.2}"
    );
    let late_mean = stats::mean(&learned_late_rewards);
    eprintln!("mean late-phase reward per decision: {late_mean:.3}");
    assert!(
        late_mean > -1.5,
        "trained agents should trade near break-even late in the run, got {late_mean:.3} per decision"
    );
}
