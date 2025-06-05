import yaml
import os
import logging
from logger import setup_logger

# Import modules
from data_loader import DataLoader
from pair_selector import PairSelector
from kalman_hedge import KalmanHedge
from signal_generator import SignalGenerator
from backtester import Backtester
from risk_engine import RiskEngine
from portfolio_optimizer import PortfolioOptimizer

# Set up root logger
logger = setup_logger("PairTradingStrategy")

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def main():
    # 1) Load configuration
    config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
    cfg = load_config(config_path)
    logger.info("Configuration loaded.")

    # 2) Fetch data
    data_cfg = cfg["data"]
    dl = DataLoader(
        tickers=data_cfg["tickers"],
        start_date=data_cfg["start_date"],
        end_date=data_cfg["end_date"],
        interval=data_cfg["interval"]
    )
    prices, volume = dl.fetch_data()

    # 3) Select pairs
    ps_cfg = cfg["pair_selector"]
    pair_selector = PairSelector(
        prices=prices,
        cluster_size=ps_cfg["cluster_size"],
        coint_pval_threshold=ps_cfg["coint_pval_threshold"],
        rolling_window=ps_cfg["rolling_window"],
        rolling_step=ps_cfg["rolling_step"],
        min_valid_periods=ps_cfg["min_valid_periods"]
    )
    pairs_df = pair_selector.select_pairs()
    if pairs_df.empty:
        logger.error("No suitable pairs found. Exiting.")
        return
    logger.info(f"Number of selected pairs: {len(pairs_df)}")

    # 4) For each selected pair, run Kalman hedge, generate signals, backtest
    all_pair_returns = {}
    results_summary = []
    for idx, row in pairs_df.iterrows():
        t1 = row["ticker1"]
        t2 = row["ticker2"]
        logger.info(f"Processing pair {t1}-{t2}.")

        s1 = prices[t1]
        s2 = prices[t2]

        # 4a) Kalman hedge
        km_cfg = cfg["kalman"]
        kh = KalmanHedge(
            observation_series=s1,
            control_series=s2,
            initial_state_cov=km_cfg["initial_state_cov"],
            transition_cov=km_cfg["transition_cov"],
            observation_cov=km_cfg["observation_cov"],
            em_iterations=km_cfg["em_iterations"]
        )
        kalman_df = kh.run_filter()

        # 4b) Signal generation
        sig_cfg = cfg["signal"]
        sg = SignalGenerator(
            price1=s1,
            price2=s2,
            kalman_df=kalman_df,
            config=sig_cfg
        )
        trade_df = sg.generate(costs=cfg["costs"], volume=volume[[t1, t2]])

        # 4c) Backtest
        bt = Backtester(
            trade_df=trade_df.rename(columns={"pos1": t1 + "_pos", "pos2": t2 + "_pos", 
                                               "price1": t1 + "_price", "price2": t2 + "_price"}),
            costs=cfg["costs"],
            volume=volume[[t1, t2]].rename(columns={t1: t1 + "_vol", t2: t2 + "_vol"})
        )
        bt_results = bt.run()
        metrics = bt.performance_metrics(bt_results)
        logger.info(f"Pair {t1}-{t2} metrics: {metrics}")

        # Store returns series for portfolio optimization
        all_pair_returns[f"{t1}/{t2}"] = bt_results["strategy_return"]

        # Summarize
        results_summary.append({
            "pair": f"{t1}/{t2}",
            **metrics,
            "half_life": row["half_life"]
        })

    # 5) Aggregate pair returns into DataFrame
    pair_returns_df = (
        pd.DataFrame(all_pair_returns)
        .dropna(how="all")
    )

    # 6) Portfolio optimization
    port_cfg = cfg["portfolio"]
    po = PortfolioOptimizer(
        pair_returns=pair_returns_df,
        min_weight=port_cfg["min_weight"],
        max_weight=port_cfg["max_weight"]
    )
    weights = po.min_variance()

    # 7) Compute portfolio P&L
    portfolio_ret = (pair_returns_df * weights).sum(axis=1)
    re = RiskEngine(returns=portfolio_ret, config=cfg["risk"])
    var_h = re.historical_var()
    var_p = re.parametric_var()
    max_dd = re.max_drawdown()
    logger.info(f"Portfolio VaR (hist) = {var_h:.4%}, (param) = {var_p:.4%}, max DD = {max_dd:.4%}")

    # 8) Save summary to CSV
    summary_df = pd.DataFrame(results_summary)
    output_dir = os.path.join(os.path.dirname(__file__), "../output")
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "pair_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved pair summary to {summary_path}.")

    weights_path = os.path.join(output_dir, "portfolio_weights.csv")
    weights.to_csv(weights_path, header=True)
    logger.info(f"Saved portfolio weights to {weights_path}.")

    logger.info("Backtest pipeline completed successfully.")

if __name__ == "__main__":
    main()
