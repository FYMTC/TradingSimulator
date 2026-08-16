//! M4 acceptance harness: the player trading loop on the live market.
//!
//! The game layer is verified the same way as every earlier layer -
//! through behaviour the design report promises, not implementation
//! details:
//!
//! - the A-share price-limit band gates every quote, edges are legal,
//!   and the band re-centres on each session close;
//! - the player's orders execute against the very same book the agents
//!   use, with T+1, funds and fills all visible in the snapshot;
//! - the manipulator runs a visible pump-and-dump arc: quiet
//!   accumulation, a marked run-up, then distribution back into the
//!   crowd - the single-player "story" the game sells;
//! - the whole session, player included, stays deterministic and
//!   replayable from one seed.

use trading_simulator::game::{GameConfig, GameMarket, ManipParams};
use trading_simulator::hetero::HeteroMarketConfig;
use trading_simulator::{Account, Exchange, ExchangeError, Money, Side};

/// Short trading days (5 sim minutes) so the manipulator's cycle crosses
/// T+1 boundaries inside a fast test.
fn game_config(seed: u64) -> GameConfig {
    GameConfig {
        market: HeteroMarketConfig {
            seed,
            day_length_ms: 300_000,
            ..HeteroMarketConfig::default()
        },
        manip: ManipParams::default(),
        ..GameConfig::default()
    }
}

fn player_funds(game: &GameMarket) -> Money {
    game.exchange()
        .account(game.player_account_id())
        .unwrap()
        .cash_available
}

#[test]
fn price_limits_gate_player_orders() {
    let mut game = GameMarket::new(game_config(11));
    // Warm the book for a few seconds so quotes reference a live mid.
    game.advance_ms(20_000);
    let snapshot = game.snapshot();
    let (lower, upper) = snapshot.price_limits.expect("limits are configured");

    // One tick outside either edge is rejected with the band attached.
    assert_eq!(
        game.submit_player_order(Side::Buy, upper + 1, 1),
        Err(ExchangeError::PriceOutsideLimits {
            price: upper + 1,
            lower,
            upper,
        })
    );
    assert_eq!(
        game.submit_player_order(Side::Sell, lower - 1, 1),
        Err(ExchangeError::PriceOutsideLimits {
            price: lower - 1,
            lower,
            upper,
        })
    );

    // In-band orders are accepted; a quote inside the band rests.
    let resting = game
        .submit_player_order(Side::Buy, snapshot.mark - 5, 1)
        .expect("in-band order is accepted");
    assert!(resting.remaining > 0);
    let snapshot = game.snapshot();
    assert_eq!(snapshot.player.open_orders.len(), 1);

    // Cancelling releases the reservation back to the player.
    let reserved_with_order = snapshot.player.cash_reserved;
    assert!(reserved_with_order > 0);
    game.cancel_player_order(resting.order_id).unwrap();
    let snapshot = game.snapshot();
    assert!(snapshot.player.open_orders.is_empty());
    assert_eq!(snapshot.player.cash_reserved, 0);
}

#[test]
fn player_round_trip_executes_with_t_plus_one() {
    // A fresh player with no seed position: everything the player owns
    // afterwards must have been bought, so T+1 is fully visible.
    let config = GameConfig {
        player_seed_shares: 0,
        ..game_config(21)
    };
    let mut game = GameMarket::new(config);
    game.advance_ms(60_000);
    let snapshot = game.snapshot();
    let ask = snapshot.best_ask.expect("agents quote both sides");

    // Cross the spread: the fill is reported inline, shares arrive
    // unsettled (T+1) and cash leaves the account.
    let cash_before = player_funds(&game);
    let buy = game
        .submit_player_order(Side::Buy, ask, 2)
        .expect("buy crosses the live market");
    assert!(!buy.trades.is_empty(), "aggressive buy fills");
    let bought: i64 = buy.trades.iter().map(|trade| trade.quantity).sum();
    assert_eq!(bought, 200);
    let snapshot = game.snapshot();
    assert_eq!(snapshot.player.unsettled_buys, 200);
    assert_eq!(snapshot.player.sellable, 0);
    assert!(player_funds(&game) < cash_before);

    // Same-day resale is blocked: T+1 applies to the player too.
    let bid = snapshot.best_bid.unwrap();
    assert!(matches!(
        game.submit_player_order(Side::Sell, bid, 1),
        Err(ExchangeError::InsufficientSellable { .. })
    ));

    // Roll over the settlement boundary, then close the position.
    game.advance_ms(300_000);
    let snapshot = game.snapshot();
    assert_eq!(snapshot.player.sellable, 200);
    let bid = snapshot.best_bid.unwrap();
    let sell = game
        .submit_player_order(Side::Sell, bid, 2)
        .expect("settled shares are sellable");
    assert!(!sell.trades.is_empty(), "sell hits the live bid");

    // Exact cash-flow conservation: equity equals cash (position flat),
    // and cash changed by exactly the traded cash flows.
    let buy_cost: i64 = buy
        .trades
        .iter()
        .map(|trade| trade.price * trade.quantity)
        .sum();
    let sell_proceeds: i64 = sell
        .trades
        .iter()
        .map(|trade| trade.price * trade.quantity)
        .sum();
    let snapshot = game.snapshot();
    assert_eq!(snapshot.player.settled, 0);
    assert_eq!(snapshot.player.unsettled_buys, 0);
    assert_eq!(snapshot.player.equity, player_funds(&game));
    assert_eq!(
        cash_before - (buy_cost as i128 - sell_proceeds as i128),
        player_funds(&game) as i128
    );
}

#[test]
fn manipulator_runs_a_pump_and_dump_arc() {
    let mut game = GameMarket::new(game_config(2026));
    let mut peak_after_pump = 0;
    let mut saw_pump = false;
    // Watch the arc in slices; the cycle spans ~850s of sim time.
    for _ in 0..34 {
        game.advance_ms(50_000);
        match game.manip_phase() {
            trading_simulator::game::ManipPhase::Pump => {
                saw_pump = true;
            }
            trading_simulator::game::ManipPhase::Distribute => {
                if saw_pump && peak_after_pump == 0 {
                    peak_after_pump = game.manip_pump_peak_mark();
                }
            }
            _ => {}
        }
    }
    eprintln!(
        "arc: pump start {} peak {} distributed {} shares, final mark {}",
        game.manip_pump_start_mark(),
        game.manip_pump_peak_mark(),
        game.manip_distributed_shares(),
        game.snapshot().mark
    );

    // The full arc played out.
    assert_eq!(
        game.manip_phase(),
        trading_simulator::game::ManipPhase::Done
    );
    assert!(saw_pump, "the manipulator must reach the pump phase");

    // The pump moved the market: peak at least 2% above the start.
    let start = game.manip_pump_start_mark();
    assert!(
        game.manip_pump_peak_mark() >= start + start / 50,
        "pump peak {} should run well above the start {}",
        game.manip_pump_peak_mark(),
        start
    );

    // Distribution handed the inventory back to the crowd.
    assert!(
        game.manip_distributed_shares() >= 5_000,
        "manipulator sold {} shares during distribution",
        game.manip_distributed_shares()
    );

    // After distribution the price fell back below the pump peak.
    let final_mark = game.snapshot().mark;
    assert!(
        final_mark < peak_after_pump,
        "final mark {final_mark} should be below the post-pump peak {peak_after_pump}"
    );
}

#[test]
fn same_seed_reproduces_the_game_exactly() {
    let run = |seed: u64| {
        let mut game = GameMarket::new(game_config(seed));
        game.advance_ms(120_000);
        let buy = game
            .submit_player_order(Side::Buy, game.snapshot().best_ask.unwrap(), 1)
            .unwrap();
        game.advance_ms(80_000);
        game.cancel_player_order(buy.order_id).ok();
        game.advance_ms(100_000);
        (
            game.tape().to_vec(),
            game.replay_log().to_vec(),
            game.snapshot().player.equity,
        )
    };
    assert_eq!(run(7), run(7));
    assert_ne!(run(7), run(8));
}

#[test]
fn replay_log_rebuilds_the_game_exchange() {
    let config = game_config(9);
    let mut game = GameMarket::new(config.clone());
    game.advance_ms(90_000);
    game.submit_player_order(Side::Buy, game.snapshot().best_ask.unwrap(), 1)
        .unwrap();
    game.advance_ms(210_000); // crosses a settlement boundary

    // Rebuild: same accounts (agents, player, manipulator), same initial
    // price-limit band, then feed the canonical log.
    let mut rebuilt = Exchange::new(config.market.symbol.clone());
    let agents = (config.market.n_noise
        + config.market.n_market_makers
        + config.market.n_trend
        + config.market.n_mean_revert
        + config.market.n_fundamental) as u64;
    for id in 0..agents {
        let mut account = Account::with_cash(config.market.agent_cash);
        account.seed_settled_position(&config.market.symbol, config.market.agent_seed_shares);
        rebuilt.add_account(id, account).unwrap();
    }
    let mut player = Account::with_cash(config.player_cash);
    player.seed_settled_position(&config.market.symbol, config.player_seed_shares);
    rebuilt.add_account(agents, player).unwrap();
    let mut manip = Account::with_cash(config.manip_cash);
    manip.seed_settled_position(&config.market.symbol, config.manip_seed_shares);
    rebuilt.add_account(agents + 1, manip).unwrap();
    rebuilt.set_price_limits(config.market.ref_price, config.limit_bp);

    rebuilt
        .replay(game.replay_log().to_vec())
        .expect("log keys are unique and ordered");
    assert_eq!(&rebuilt, game.exchange());
}
