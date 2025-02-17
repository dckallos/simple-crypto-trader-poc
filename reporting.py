import sqlite3
import time
import logging
import datetime
from collections import defaultdict
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class AnalyticsReporter:
    """
    A production-ready class for analyzing PnL, trade effectiveness,
    and other KPIs based on ledger_entries + trades data in your SQLite DB.

    We keep the original logic, and add:
      - enrich_details_with_pair(details) => fetch pair from `trades` table
      - compute_pnl_breakdown_by_pair(details) => returns {pair: {...stats...}}
    """

    def __init__(self, db_path: str = "trades.db", lookback_hours: int = 24):
        """
        :param db_path: path to your SQLite DB with 'ledger_entries' and 'trades'
        :param lookback_hours: default lookback for compute_24h_pnl() or generate_report()
        """
        self.db_path = db_path
        self.lookback_hours = lookback_hours

    def compute_24h_pnl(self) -> Dict[str, Any]:
        """
        Convenience method => returns a dict with:
         {
            'total_realized_pnl': <float>,
            'closed_trades_count': <int>,
            'details': [ ... ]
         }
        for all trades in the last 24 hours.
        """
        now_ts = int(time.time())
        start_ts = now_ts - (24 * 3600)  # 24 hours ago
        return self.calculate_period_pnl(start_ts, now_ts)

    def calculate_period_pnl(self, start_ts: int, end_ts: int) -> Dict[str, Any]:
        """
        Main method that:
         1) Loads ledger_entries in ascending order (type='trade', time in [start_ts, end_ts]).
         2) Groups them by 'refid'.
         3) For each refid, identifies buy vs sell, calculates cost-basis or realized PnL
            with a weighted-average cost approach.
         4) Returns a summary dict with 'total_realized_pnl', 'closed_trades_count', 'details'.

        We'll later call enrich_details_with_pair(...) so that the 'details' also
        know the actual 'pair' from the `trades` table.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            # 1) fetch ledger trades in ascending time order
            c = conn.cursor()
            sql = """
            SELECT ledger_id, refid, time, type, asset, amount, fee
            FROM ledger_entries
            WHERE type='trade'
              AND time BETWEEN ? AND ?
            ORDER BY time ASC, ledger_id ASC
            """
            rows = c.execute(sql, (start_ts, end_ts)).fetchall()
            if not rows:
                return {
                    'total_realized_pnl': 0.0,
                    'closed_trades_count': 0,
                    'details': []
                }

            # 2) group by refid
            grouped = defaultdict(list)
            for r in rows:
                grouped[r['refid']].append(dict(r))

            # costBasis[ 'ADA' ] = ( quantity_held_in_ada, average_cost_usd_per_ada )
            costBasis = {}
            details = []
            total_pnl = 0.0
            closed_trades_count = 0

            # We'll process each refid in ascending time order => stable cost-basis approach
            refid_time_pairs = []
            for rf, items in grouped.items():
                earliest_time = min(x['time'] for x in items)
                refid_time_pairs.append((rf, earliest_time))
            refid_time_pairs.sort(key=lambda x: x[1])  # sort by earliest_time

            for (rf, earliest_time) in refid_time_pairs:
                items = grouped[rf]
                # Summation approach => find the base asset line and the zusd line
                base_asset = None
                base_amount = 0.0
                zusd_amount = 0.0
                total_fee_usd = 0.0
                trade_ts = max(x['time'] for x in items)

                for it in items:
                    asset = it['asset']
                    amt = float(it['amount'] or 0.0)
                    fee = float(it['fee'] or 0.0)
                    if asset.upper() == 'ZUSD':
                        zusd_amount += amt
                        total_fee_usd += fee
                    else:
                        if base_asset is None:
                            base_asset = asset
                        elif base_asset != asset:
                            logger.warning(
                                f"Refid={rf} has multiple base assets? {base_asset} vs {asset}"
                            )
                            continue
                        base_amount += amt

                if not base_asset:
                    continue

                side = 'BUY' if base_amount > 0 else 'SELL'
                qty = abs(base_amount)
                if qty < 1e-12:
                    logger.debug(f"Refid={rf} => ignoring zero-qty trade.")
                    continue

                gross_zusd = abs(zusd_amount)
                gross_price = gross_zusd / qty

                realized_pnl = 0.0
                asset_key = base_asset.upper()
                old_qty, old_cost = costBasis.get(asset_key, (0.0, 0.0))

                if side == 'BUY':
                    # cost of buy = abs(zusd_amount) + total_fee_usd
                    total_cost_for_buy = abs(zusd_amount) + total_fee_usd
                    new_qty = old_qty + qty
                    if new_qty > 0:
                        new_cost = (old_qty * old_cost + total_cost_for_buy) / new_qty
                    else:
                        new_cost = 0.0
                    costBasis[asset_key] = (new_qty, new_cost)

                else:  # SELL
                    net_proceeds = abs(zusd_amount) - total_fee_usd
                    if old_qty < 1e-12:
                        logger.warning(
                            f"Refid={rf} => SELL {qty} {asset_key}, but no position => PnL=proceeds only."
                        )
                        realized_pnl = net_proceeds
                    else:
                        if qty > old_qty:
                            logger.warning(
                                f"Refid={rf} => SELL {qty} {asset_key} but only {old_qty} in costBasis."
                            )
                            used_qty = old_qty
                            leftover_qty = 0.0
                        else:
                            used_qty = qty
                            leftover_qty = old_qty - qty

                        cost_for_this_sell = used_qty * old_cost
                        realized_pnl = net_proceeds - cost_for_this_sell

                        if leftover_qty < 1e-12:
                            costBasis[asset_key] = (0.0, 0.0)
                        else:
                            costBasis[asset_key] = (leftover_qty, old_cost)

                    total_pnl += realized_pnl
                    closed_trades_count += 1

                details.append({
                    'refid': rf,
                    'asset': base_asset,
                    'side': side,
                    'quantity': qty,
                    'price': gross_price,
                    'usd_fee': total_fee_usd,
                    'pnl': realized_pnl,
                    'timestamp': trade_ts
                })

            return {
                'total_realized_pnl': total_pnl,
                'closed_trades_count': closed_trades_count,
                'details': details
            }

        finally:
            conn.close()

    def compute_trade_effectiveness(self, pnl_results: Dict[str, Any]) -> Dict[str, float]:
        """
        Given the dict from `calculate_period_pnl` or `compute_24h_pnl`,
        compute how many sells had positive vs negative PnL, average wins, etc.
        """
        details = pnl_results.get('details', [])
        sell_trades = [d for d in details if d['side'] == 'SELL']
        if not sell_trades:
            return {
                'winning_trades': 0,
                'losing_trades': 0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'win_rate': 0.0
            }

        winners = [t for t in sell_trades if t['pnl'] > 1e-12]
        losers = [t for t in sell_trades if t['pnl'] < -1e-12]

        winning_trades = len(winners)
        losing_trades = len(losers)
        total_sells = len(sell_trades)

        avg_win = sum(t['pnl'] for t in winners) / winning_trades if winning_trades else 0.0
        avg_loss = sum(t['pnl'] for t in losers) / losing_trades if losing_trades else 0.0
        win_rate = winning_trades / total_sells if total_sells else 0.0

        return {
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_rate': win_rate
        }

    def enrich_details_with_pair(self, details: List[dict]) -> None:
        """
        For each entry in 'details', look up the `pair` from the 'trades' table,
        matching details[i]['refid'] => trades.order_id. We store it as details[i]['pair'].

        This way, we can do per-pair breakdown of realized PnL.

        In practice, we only need this for SELL trades to track realized PnL,
        but we'll fill 'pair' for all if found.
        """
        if not details:
            return

        # Grab all distinct refids from the details
        refids = set(d['refid'] for d in details)
        if not refids:
            return

        placeholders = ",".join("?" for _ in refids)  # for SQL IN clause
        refid_list = list(refids)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            c = conn.cursor()
            sql = f"""
                SELECT order_id, pair
                FROM trades
                WHERE order_id IN ({placeholders})
            """
            rows = c.execute(sql, refid_list).fetchall()
            # Build a dict refid->pair
            refid_to_pair = {}
            for r in rows:
                refid_to_pair[r['order_id']] = r['pair']

            # Now attach to each details row
            for d in details:
                rr = d['refid']
                the_pair = refid_to_pair.get(rr)
                if the_pair:
                    d['pair'] = the_pair
                else:
                    d['pair'] = None

        finally:
            conn.close()

    def compute_pnl_breakdown_by_pair(self, details: List[dict]) -> Dict[str, Dict[str, float]]:
        """
        After we have 'details' that contain 'side' == 'SELL' with realized PnL
        and have been enriched with 'pair', we can group net PnL by pair.

        Returns a dict like:
           {
             "ETH/USD": {
                "pnl": 12.38,
                "trades": 3,
                "wins": 2,
                "losses": 1
             },
             "ADA/USD": {...}
           }
        """
        pair_map = defaultdict(lambda: {"pnl": 0.0, "trades": 0, "wins": 0, "losses": 0})
        for d in details:
            if d.get("side") == "SELL":
                pair = d.get("pair")
                if not pair:
                    # Might be None if trades table had no record
                    continue
                pair_map[pair]["pnl"] += d["pnl"]
                pair_map[pair]["trades"] += 1
                if d["pnl"] > 0:
                    pair_map[pair]["wins"] += 1
                elif d["pnl"] < 0:
                    pair_map[pair]["losses"] += 1
        return dict(pair_map)

    def generate_report(self, lookback_hours: int = None) -> Dict[str, Any]:
        """
        High-level method to produce a combined summary for the chosen lookback.
        Now includes per-pair breakdown of realized PnL.

        Example returned structure:
          {
            'pnl_summary': {...},            # from calculate_period_pnl
            'trade_effectiveness': {...},    # from compute_trade_effectiveness
            'pair_breakdown': { ... }        # from compute_pnl_breakdown_by_pair
          }
        """
        if lookback_hours is None:
            lookback_hours = self.lookback_hours

        now_ts = int(time.time())
        start_ts = now_ts - (lookback_hours * 3600)
        pnl_res = self.calculate_period_pnl(start_ts, now_ts)

        # Enrich with actual 'pair' from trades table
        self.enrich_details_with_pair(pnl_res['details'])

        # Overall trade effectiveness
        eff_res = self.compute_trade_effectiveness(pnl_res)

        # Per-pair breakdown
        pair_breakdown = self.compute_pnl_breakdown_by_pair(pnl_res['details'])

        return {
            'pnl_summary': pnl_res,
            'trade_effectiveness': eff_res,
            'pair_breakdown': pair_breakdown
        }

    def print_report(self, lookback_hours: int = None):
        """
        Convenience: fetch the combined report and print it in a human-readable form.
        Includes the per-pair breakdown of realized PnL.
        """
        if lookback_hours is None:
            lookback_hours = self.lookback_hours

        rep = self.generate_report(lookback_hours)
        pnl_summ = rep['pnl_summary']
        eff_summ = rep['trade_effectiveness']
        pair_summ = rep['pair_breakdown']

        print(f"--- PnL Summary (last {lookback_hours}h) ---")
        print(f"Total Realized PnL: {pnl_summ['total_realized_pnl']:.4f} USD")
        print(f"Closed Trades (SELLs): {pnl_summ['closed_trades_count']}")

        print(f"\n--- Trade Effectiveness ---")
        print(f"Winners: {eff_summ['winning_trades']}   Losers: {eff_summ['losing_trades']}")
        print(f"Win Rate: {eff_summ['win_rate'] * 100:.2f}%")
        print(f"Avg Win:  {eff_summ['avg_win']:.4f}   Avg Loss: {eff_summ['avg_loss']:.4f}")

        print(f"\n--- Pair Breakdown of Realized PnL (SELL trades) ---")
        if not pair_summ:
            print("No sells found => No per‐pair data.")
        else:
            for pair, stats in pair_summ.items():
                print(f"  {pair}: PnL={stats['pnl']:.4f}  sells={stats['trades']}  "
                      f"wins={stats['wins']}  losses={stats['losses']}")

        print(f"\n(Use pnl_summary['details'] for each SELL trade’s PnL breakdown.)")

if __name__ == "__main__":
    reporter = AnalyticsReporter()
    reporter.print_report()