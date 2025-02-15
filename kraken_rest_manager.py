#!/usr/bin/env python3
# =============================================================================
# FILE: kraken_rest_manager.py
# =============================================================================
"""
kraken_rest_manager.py

A unified class for Kraken REST API calls, consolidating:
  - Public endpoints (AssetPairs, etc.)
  - Private endpoints (Balance, Ledgers, WebSockets token)

Usage:
    manager = KrakenRestManager(api_key="...", api_secret="...")
    balances = manager.fetch_balance()
    manager.fetch_and_store_ledger(asset="all", ledger_type="all", db_path="trades.db")
    manager.fetch_public_asset_pairs(pair_list=["ETH/USD","XBT/USD"])
    token_json = manager.get_websocket_token()

This avoids duplication. You can rename or expand the methods.
By default, we show storing ledger entries & asset pairs in local DB if you choose to do so.
"""

import time
import hmac
import hashlib
import base64
import requests
import logging
import json
import sqlite3
from typing import Optional, Dict, Any, List

from numpy.ma.core import zeros_like

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class KrakenRestManager:
    """
    A robust class encapsulating Kraken REST calls for both public and private endpoints.

    Basic Flow:
      - Construct with your API key & secret.
      - call fetch_balance(...) for private GET /0/private/Balance
      - call fetch_and_store_ledger(...) for private GET /0/private/Ledgers
      - call fetch_public_asset_pairs(...) for public GET /0/public/AssetPairs
      - call get_websocket_token(...) to retrieve a token for wss://ws-auth.kraken.com

    If you want more endpoints, add more methods below.
    You can rename or reorganize as you see fit.
    """

    def __init__(self, api_key: str, api_secret: str):
        """
        :param api_key: The Kraken API key (with correct permissions for the calls you want).
        :param api_secret: The Kraken secret in base64 format.
        """
        self.api_key = api_key
        self.api_secret = api_secret

    # --------------------------------------------------------------------------
    # PUBLIC UTILS
    # --------------------------------------------------------------------------
    def fetch_public_asset_pairs(self, pair_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calls GET /0/public/AssetPairs to retrieve public metadata for the given pairs.
        If pair_list is None, fetches all pairs. Returns the JSON result dict.
        Example usage:
            result = manager.fetch_public_asset_pairs(pair_list=["ETH/USD","XBT/USD"])
            # store result in DB or parse it
        """
        url = "https://api.kraken.com/0/public/AssetPairs"
        params = {}
        if pair_list:
            joined = ",".join(pair_list)
            params["pair"] = joined
        logger.info(f"[KrakenRestManager] GET => {url}, params={params}")
        resp = self._public_request(url, params=params)
        return resp

    def fetch_asset_info(self, asset_list: List[str], aclass: str = "currency") -> Dict[str, Any]:
        """
        Calls GET /0/public/Assets to retrieve asset info for the given asset_list,
        e.g. ["XETH","XXBT"]. Returns the raw JSON dict from Kraken:
          {
             "error": [],
             "result": {
               "XETH": { ... },
               "XXBT": { ... }
             }
          }

        If asset_list is empty or None => retrieves all available assets.
        """
        url = "https://api.kraken.com/0/public/Assets"
        params = {}
        if asset_list:
            joined = ",".join(asset_list)
            params["asset"] = joined
        if aclass:
            params["aclass"] = aclass

        logger.info(f"[KrakenRestManager] fetch_asset_info => GET {url}, params={params}")
        resp = self._public_request(url, params=params)
        return resp  # e.g. {"error":[],"result":{...}}

    def build_coin_name_lookup_from_db(
            self,
            db_path: str = "trades.db",
            lookup_table: str = "kraken_asset_name_lookup"
    ):
        """
        1) Reads 'wsname', 'base', 'pair_name', and 'alternative_name' from 'kraken_asset_pairs' in 'db_path'.
        2) Groups rows by unique 'base' (e.g. "XETH","XXBT").
        3) For each base => calls self.fetch_asset_info([base]) => parse altname, decimals, etc.
        4) Inserts into a new or existing 'asset_name_lookup' table, using 'wsname' as unique key,
           storing:
             wsname (PK),
             base_asset,
             pair_name,
             alternative_name,
             formatted_base_asset_name (the altname from the REST response),
             aclass,
             decimals,
             display_decimals,
             collateral_value,
             status
        """
        import sqlite3
        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()
            # 1) Create asset_name_lookup table if not exists
            c.execute(f"""
            CREATE TABLE IF NOT EXISTS {lookup_table} (
                wsname TEXT PRIMARY KEY, 
                base_asset TEXT,
                pair_name TEXT,
                alternative_name TEXT,
                formatted_base_asset_name TEXT,
                aclass TEXT,
                decimals INTEGER,
                display_decimals INTEGER,
                collateral_value REAL,
                status TEXT
            )
            """)
            conn.commit()
        except Exception as e:
            logger.exception(f"[KrakenRestManager] Error creating {lookup_table} => {e}")
            conn.close()
            return

        # 2) Read wsname, base, pair_name, alternative_name from kraken_asset_pairs
        try:
            z_usd = "ZUSD"
            c.execute(f"""
                SELECT wsname, base, pair_name, altname as alternative_name
                FROM kraken_asset_pairs
                WHERE base IS NOT NULL
                AND quote = '{z_usd}'
            """)
            pair_data = c.fetchall()
        except Exception as e:
            logger.exception(f"[KrakenRestManager] Error reading kraken_asset_pairs => {e}")
            conn.close()
            return

        # We'll hold them in a dict => {base_asset: [(wsname, pair_name, altname_in_db), ...]}
        # because multiple wsnames might share the same base (e.g. "XETH")
        base_dict = {}
        for (wsname, base_val, pair_name, alt_in_db) in pair_data:
            if not base_val:
                continue
            if base_val not in base_dict:
                base_dict[base_val] = []
            base_dict[base_val].append((wsname, pair_name, alt_in_db))

        # 3) For each unique base => call fetch_asset_info([base_val])
        for base_asset, row_list in base_dict.items():
            asset_resp = self.fetch_asset_info([base_asset])  # e.g. ["XETH"]
            if not asset_resp or "result" not in asset_resp:
                logger.warning(f"[KrakenRestManager] No result for base_asset={base_asset}")
                continue

            asset_info_dict = asset_resp["result"].get(base_asset, None)
            if not asset_info_dict:
                logger.warning(f"[KrakenRestManager] No info for base_asset={base_asset} in result.")
                continue

            # parse relevant fields
            aclass = asset_info_dict.get("aclass", "")
            altname = asset_info_dict.get("altname", "")
            decimals = asset_info_dict.get("decimals", 0)
            disp_decimals = asset_info_dict.get("display_decimals", 0)
            collateral_val = asset_info_dict.get("collateral_value", None)
            status_val = asset_info_dict.get("status", "")

            # 4) For each row that references this base_asset => insert/replace in asset_name_lookup
            for (ws, kraken_pair_name, alt_in_db) in row_list:
                try:
                    c.execute(f"""
                    INSERT OR REPLACE INTO {lookup_table} (
                        wsname, 
                        base_asset,
                        pair_name,
                        alternative_name,
                        formatted_base_asset_name,
                        aclass,
                        decimals,
                        display_decimals,
                        collateral_value,
                        status
                    )
                    VALUES (?,?,?,?,?,?,?,?,?,?)
                    """, (
                        ws,
                        base_asset,
                        kraken_pair_name,
                        alt_in_db,
                        altname,  # e.g. "ETH" or "XBT" from the REST
                        aclass,
                        decimals,
                        disp_decimals,
                        float(collateral_val) if collateral_val else None,
                        status_val
                    ))
                except Exception as e:
                    logger.exception(f"[KrakenRestManager] DB insert error => {e}")

        conn.commit()
        conn.close()
        logger.info(f"[KrakenRestManager] build_coin_name_lookup_from_db => updated {lookup_table} table.")

    def store_asset_pairs_in_db(self, pairs_json: Dict[str, Any], db_path: str = "trades.db"):
        """
        Given the 'result' dict from fetch_public_asset_pairs(...),
        store or upsert them into 'kraken_asset_pairs' table.
        Example usage after you call fetch_public_asset_pairs.
        """
        from db import store_kraken_asset_pair_info

        if not pairs_json or "result" not in pairs_json:
            logger.warning("[KrakenRestManager] no 'result' in pairs_json => skip storing.")
            return

        results = pairs_json["result"]
        count = 0
        for pair_name, pair_info in results.items():
            store_kraken_asset_pair_info(pair_name, pair_info)
            count += 1
        logger.info(f"[KrakenRestManager] Upserted {count} asset pairs into DB.")

    # --------------------------------------------------------------------------
    # PRIVATE UTILS
    # --------------------------------------------------------------------------
    def fetch_balance(self) -> Dict[str, float]:
        """
        Calls /0/private/Balance => returns a dict {asset: float_balance},
        and also stores them into the DB via store_kraken_balances(...).

        The rest of your logic is the same, but now we incorporate DB storage.
        """
        endpoint = "/0/private/Balance"
        payload = {}
        result = self._private_request(endpoint, payload)
        if not result or "result" not in result:
            return {}
        raw_dict = result["result"]

        out = {}
        for k, v in raw_dict.items():
            try:
                out[k] = float(v)
            except:
                out[k] = 0.0

        logger.info(f"[KrakenRestManager] fetch_balance => {out}")

        # --- NEW: store in DB so we have a historical record ---
        from db import store_kraken_balances
        store_kraken_balances(out)  # uses default DB_FILE path

        return out

    def fetch_and_store_ledger(
        self,
        asset: str = None,
        ledger_type: str = "all",
        start: int = None,
        end: int = None,
        db_path: str = "trades.db"
    ):
        """
        Calls /0/private/Ledgers => upserts each ledger entry into 'ledger_entries' table.

        Example usage:
            manager.fetch_and_store_ledger(asset="all", ledger_type="all", db_path="trades.db")
        """
        endpoint = "/0/private/Ledgers"
        payload = {"type": ledger_type}
        if asset:
            payload["asset"] = asset
        if start:
            payload["start"] = start
        if end:
            payload["end"] = end

        resp = self._private_request(endpoint, payload)
        if not resp or "result" not in resp:
            logger.warning("[KrakenRestManager] fetch_and_store_ledger => no result => skip.")
            return
        ledger_dict = resp["result"].get("ledger", {})
        if not ledger_dict:
            logger.info("[KrakenRestManager] no ledger entries returned.")
            return

        conn = sqlite3.connect(db_path)
        rows_inserted = 0
        try:
            c = conn.cursor()
            for ledger_id, entry_obj in ledger_dict.items():
                refid = entry_obj.get("refid", "")
                time_val = float(entry_obj.get("time", 0.0))
                ltype = entry_obj.get("type", "")
                subtype = entry_obj.get("subtype", "")
                asset_val = entry_obj.get("asset", "")
                amt = float(entry_obj.get("amount", 0.0))
                fee_val = float(entry_obj.get("fee", 0.0))
                bal = float(entry_obj.get("balance", 0.0))

                c.execute("""
                    INSERT OR REPLACE INTO ledger_entries (
                        ledger_id, refid, time, type, subtype, asset, amount, fee, balance
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    ledger_id,
                    refid,
                    time_val,
                    ltype,
                    subtype,
                    asset_val,
                    amt,
                    fee_val,
                    bal
                ))
                rows_inserted += 1
            conn.commit()
            logger.info(f"[KrakenRestManager] Upserted {rows_inserted} ledger entries.")
        except Exception as e:
            logger.exception(f"[KrakenRestManager] DB error => {e}")
        finally:
            conn.close()

    def get_websocket_token(self) -> Optional[Dict[str, Any]]:
        """
        Calls /0/private/GetWebSocketsToken => returns JSON containing the 'token' used
        for wss://ws-auth.kraken.com. If error, returns None.

        Example usage:
            token_json = manager.get_websocket_token()
            if token_json and 'result' in token_json:
                token = token_json['result']['token']
        """
        endpoint = "/0/private/GetWebSocketsToken"
        resp = self._private_request(endpoint, {})
        if not resp or "result" not in resp:
            logger.warning("[KrakenRestManager] get_websocket_token => no result => skip.")
            return None
        return resp

    # --------------------------------------------------------------------------
    # INTERNAL: SIGNED PRIVATE REQUEST
    # --------------------------------------------------------------------------
    def _private_request(self, endpoint: str, payload: dict) -> Dict[str, Any]:
        """
        Creates a nonce, encodes payload, and POSTs to "https://api.kraken.com" + endpoint
        with your API-Key, API-Sign headers. Returns the JSON response as a dict.
        If error is returned, logs & returns an empty dict.
        """
        url = "https://api.kraken.com" + endpoint
        nonce_val = str(int(time.time() * 1000))
        payload["nonce"] = nonce_val
        import urllib.parse
        postdata = urllib.parse.urlencode(payload)
        # signature
        sha256_digest = hashlib.sha256((nonce_val + postdata).encode("utf-8")).digest()
        message = endpoint.encode("utf-8") + sha256_digest
        secret = base64.b64decode(self.api_secret)
        sig = hmac.new(secret, message, hashlib.sha512)
        signature = base64.b64encode(sig.digest()).decode()

        headers = {
            "API-Key": self.api_key,
            "API-Sign": signature,
            "Content-Type": "application/x-www-form-urlencoded"
        }

        try:
            resp = requests.post(url, headers=headers, data=postdata, timeout=10)
            resp.raise_for_status()
            j = resp.json()
            if j.get("error"):
                logger.error(f"[KrakenRestManager] private error => {j['error']}")
            return j
        except requests.exceptions.RequestException as e:
            logger.exception(f"[KrakenRestManager] private request error => {e}")
            return {}
        except Exception as e:
            logger.exception(f"[KrakenRestManager] unknown error => {e}")
            return {}

    # --------------------------------------------------------------------------
    # INTERNAL: PUBLIC REQUEST
    # --------------------------------------------------------------------------
    def _public_request(self, url: str, params: dict = None) -> Dict[str, Any]:
        """
        Simple GET for public endpoints. Returns the JSON as a dict, logs if error.
        """
        if not params:
            params = {}
        try:
            r = requests.get(url, params=params, timeout=10)
            r.raise_for_status()
            j = r.json()
            if j.get("error"):
                logger.error(f"[KrakenRestManager] public error => {j['error']}")
            return j
        except requests.exceptions.RequestException as e:
            logger.exception(f"[KrakenRestManager] public GET => {e}")
            return {}
        except Exception as e:
            logger.exception(f"[KrakenRestManager] unknown => {e}")
            return {}


if __name__ == "__main__":
    import os
    import json
    from dotenv import load_dotenv
    load_dotenv()
    manager = KrakenRestManager(api_key=os.getenv("KRAKEN_API_KEY"), api_secret=os.getenv("KRAKEN_SECRET_API_KEY") )
    print(f"balance: {json.dumps(manager.fetch_balance(), indent=4)}")
    manager.fetch_and_store_ledger(db_path="trades.db")
    print(f"Public asset pairs: {json.dumps(manager.fetch_public_asset_pairs(["ETH/USD"]), indent=4)}")
    manager.fetch_public_asset_pairs(["ETH/USD"])
    token_json = manager.get_websocket_token()
    print(f"token = {token_json}")
    manager.build_coin_name_lookup_from_db()

