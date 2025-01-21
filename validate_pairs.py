#!/usr/bin/env python3
"""
validate_pairs.py

Loads 'config.yaml' from the current directory, fetches the list of asset pairs from Kraken,
and prints which pairs in config.yaml are recognized by Kraken vs. which are not.

Usage:
    python validate_pairs.py
"""

import os
import requests
import yaml

def load_config_pairs(config_file="config.yaml"):
    """
    Loads the traded_pairs list from config.yaml.
    :param config_file: path to the YAML file.
    :return: list of pairs (e.g., ["XBT/USD", "ETH/USD"]).
    """
    if not os.path.exists(config_file):
        print(f"Config file {config_file} not found.")
        return []

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config.get("traded_pairs", [])


def fetch_kraken_asset_pairs():
    """
    Fetches the full list of Kraken asset pairs from the public REST endpoint.
    Returns a dict of pairName -> pairInfo, for example:
        {
          "XXBTZUSD": {...},
          "XETHZUSD": {...},
          ...
        }
    or an empty dict if there's an error.
    """
    url = "https://api.kraken.com/0/public/AssetPairs"
    try:
        resp = requests.get(url, timeout=10)
        data = resp.json()
        # If there's an error, data["error"] might have info
        if data.get("error"):
            print(f"Kraken error: {data['error']}")
            return {}
        return data.get("result", {})
    except Exception as e:
        print(f"Exception fetching asset pairs: {e}")
        return {}


def friendly_to_kraken_format(friendly_pair: str) -> str:
    """
    Tries to transform "XBT/USD" to "XBTUSD" (remove slash).
    This is a naive approach. Real mapping might require e.g. "XBT/USD" -> "XXBTZUSD".
    We'll just remove '/' and see if it partially matches any known Kraken pair key.
    """
    return friendly_pair.replace("/", "")


def main():
    # 1. Load pairs from config.yaml
    config_pairs = load_config_pairs()
    if not config_pairs:
        print("No traded_pairs found in config.yaml.")
        return

    # 2. Fetch the official Kraken asset pairs
    kraken_pairs = fetch_kraken_asset_pairs()
    if not kraken_pairs:
        print("Could not fetch Kraken asset pairs, exiting.")
        return

    # 3. For each pair in config.yaml, try to see if we can match it in kraken_pairs
    valid_pairs = []
    invalid_pairs = []

    kraken_keys = list(kraken_pairs.keys())  # e.g. ["XXBTZUSD", "XETHZUSD", ...]
    for friendly_pair in config_pairs:
        transformed = friendly_to_kraken_format(friendly_pair)  # e.g. "XBT/USD" -> "XBTUSD"
        # Check if that transformed string is in any of the Kraken pair keys
        # or if we can do a partial match.
        matched_key = None
        for key in kraken_keys:
            if transformed.upper() in key.upper():
                matched_key = key
                break

        if matched_key:
            valid_pairs.append((friendly_pair, matched_key))
        else:
            invalid_pairs.append(friendly_pair)

    # 4. Print results
    print("\nVALID PAIRS (config -> kraken key):")
    for friendly, kraken_key in valid_pairs:
        print(f"  {friendly} => {kraken_key}")

    print("\nINVALID (not found on Kraken):")
    for p in invalid_pairs:
        print(f"  {p}")


if __name__ == "__main__":
    main()
