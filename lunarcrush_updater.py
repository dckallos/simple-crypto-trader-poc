#!/usr/bin/env python3
"""
lunarcrush_updater.py

A script that provides an easy way to update and backfill LunarCrush data on
a configurable schedule. It consumes and calls logic from fetch_lunarcrush.py,
which holds the LunarCrushFetcher class and methods.

Usage Examples:
    1) Single run: Snapshot + time-series update
       $ python lunarcrush_updater.py --snapshot --time-series

    2) Single run: Backfill 12 months
       $ python lunarcrush_updater.py --backfill --backfill-months 12

    3) Continuous updates every 60 minutes:
       $ python lunarcrush_updater.py --continuous --interval-minutes 60 --snapshot

    4) Continuous backfill (careful with large intervals; you may want separate tasks):
       $ python lunarcrush_updater.py --continuous --interval-minutes 180 --backfill --backfill-months 3
"""

import time
import logging
import argparse

# If fetch_lunarcrush is in the same folder or is importable
from fetch_lunarcrush import LunarCrushFetcher, DB_FILE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """
    Returns command line arguments for controlling the LunarCrush updates.
    """
    parser = argparse.ArgumentParser(description="Update and backfill LunarCrush data.")
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Perform a snapshot update (fetch top coins list from LunarCrush and store in DB)."
    )
    parser.add_argument(
        "--snapshot-filtered",
        action="store_true",
        help="Perform a snapshot update but only for specific filter_symbols."
    )
    parser.add_argument(
        "--time-series",
        action="store_true",
        help="Fetch time-series data for all symbols found in 'lunarcrush_data'."
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Perform a chunk-based backfill for all symbols found in 'lunarcrush_data'."
    )
    parser.add_argument(
        "--backfill-months",
        type=int,
        default=6,
        help="How many months to backfill when --backfill is used. Default=6."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Limit for snapshot calls. Default=100."
    )
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run the selected tasks in an infinite loop, sleeping between runs."
    )
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=60,
        help="How many minutes to wait between continuous runs. Default=60."
    )

    return parser.parse_args()


def perform_updates(args):
    """
    Performs the requested updates once, based on command-line arguments.
    """
    fetcher = LunarCrushFetcher(db_file=DB_FILE)

    # Snapshot (unfiltered or filtered)
    if args.snapshot_filtered:
        logger.info("Performing snapshot for only filter_symbols...")
        fetcher.fetch_snapshot_data_filtered(limit=args.limit)
    elif args.snapshot:
        logger.info("Performing snapshot for all coins...")
        fetcher.fetch_snapshot_data_all_coins(limit=args.limit)

    # Time-series fetch for all known symbols
    if args.time_series:
        logger.info("Performing time-series fetch for tracked symbols...")
        fetcher.fetch_time_series_for_tracked_coins(bucket="hour", interval="1w")

    # Backfill chunk-based
    if args.backfill:
        logger.info(f"Performing backfill for the last {args.backfill_months} months...")
        # Load all symbols from the DB
        coin_ids = fetcher._load_symbols_from_db()  # or a public method if you prefer
        if not coin_ids:
            logger.warning("No coin symbols found in DB to backfill.")
        else:
            fetcher.backfill_coins(
                coin_ids=coin_ids,
                months=args.backfill_months,
                bucket="hour",
                interval="1w"
            )


def main():
    args = parse_args()

    if args.continuous:
        logger.info(
            "Running in continuous mode. "
            f"Will repeat updates every {args.interval_minutes} minutes."
        )
        while True:
            perform_updates(args)
            logger.info(f"Completed an update cycle. Sleeping {args.interval_minutes} minutes.")
            time.sleep(args.interval_minutes * 60)  # convert minutes to seconds
    else:
        # Single pass
        perform_updates(args)


if __name__ == "__main__":
    main()
