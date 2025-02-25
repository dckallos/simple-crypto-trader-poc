#!/usr/bin/env python3

# =============================================================================
# FILE: kraken_rest_manager.py
# =============================================================================
"""
kraken_rest_manager.py (Final, bulk-fetched build_coin_name_lookup_from_db)

This version avoids calling fetch_asset_info(...) repeatedly for each base asset,
which previously caused many timeouts. Instead, we gather all base assets, chunk
them if needed, do fewer requests, then parse them at once.

Additionally, _public_request now uses a longer default timeout (30s).
"""

import time
import hmac
import hashlib
import base64
import requests
import logging
import sqlite3
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class KrakenRestManager:
    """
    A robust class encapsulating Kraken REST calls for both public and private endpoints.
    """

    def __init__(self, api_key: str, api_secret: str):
        """
        :param api_key: The Kraken API key (with correct permissions for the calls you want).
        :param api_secret: The Kraken secret in base64 format.
        """
        self.api_key = api_key
        self.api_secret = api_secret

    # --------------------------------------------------------------------------
    # PUBLIC: GET /0/public/AssetPairs
    # --------------------------------------------------------------------------
    def fetch_public_asset_pairs(self, pair_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calls GET /0/public/AssetPairs to retrieve public metadata for the given pairs.
        If pair_list is None, fetches all pairs. Returns the JSON result dict.
        """
        url = "https://api.kraken.com/0/public/AssetPairs"
        params = {}
        if pair_list:
            joined = ",".join(pair_list)
            params["pair"] = joined
        logger.info(f"[KrakenRestManager] GET => {url}, params={params}")
        resp = self._public_request(url, params=params)
        return resp

    def convert_wsnames_to_pair_names(self, wsnames: List[str]) -> List[str]:
        """
        Converts a list of WebSocket names (e.g., "ETH/USD") to Kraken-specific pair names (e.g., "XETHZUSD").

        Args:
            wsnames (List[str]): List of WebSocket names, e.g., ["ETH/USD", "XBT/USD"].

        Returns:
            List[str]: List of corresponding Kraken pair names, e.g., ["XETHZUSD", "XXBTZUSD"].
        """
        from db_lookup import get_asset_value_for_pair

        pair_names = []
        for wsname in wsnames:
            pair_name = get_asset_value_for_pair(wsname, "pair_name")
            if pair_name:
                pair_names.append(pair_name)
            else:
                logger.warning(f"[KrakenRestManager] No pair_name found for wsname={wsname}, skipping.")
        return pair_names

    def fetch_and_store_trade_volume(self, wsnames: List[str], db_path: str = "trades.db"):
        """
        Fetches trade volume and fee information for the given WebSocket names and stores it in the kraken_trading_fees table.

        Args:
            wsnames (List[str]): List of asset pairs in WebSocket format, e.g., ["ETH/USD", "XBT/USD"].
            db_path (str): Path to the SQLite database (default: "trades.db").

        Raises:
            ValueError: If no WebSocket names are provided.
        """
        if not wsnames:
            raise ValueError("WebSocket names must be provided to fetch fee information.")

        # Convert WebSocket names to Kraken-specific pair names
        pair_names = self.convert_wsnames_to_pair_names(wsnames)
        if not pair_names:
            logger.error("[KrakenRestManager] No valid pair_names after conversion, aborting.")
            return

        # Generate nonce for the API request
        nonce_val = str(int(time.time() * 1000))

        # Create payload with comma-separated pair names
        payload = {
            "nonce": nonce_val,
            "pair": ",".join(pair_names)
        }

        # Make the private API request
        endpoint = "/0/private/TradeVolume"
        response = self._private_request(endpoint, payload)  # Assume this method exists

        # Validate the response
        if not response or "result" not in response:
            logger.error("[KrakenRestManager] fetch_and_store_trade_volume => no 'result' in response.")
            return

        result = response["result"]
        currency = result.get("currency", "")  # e.g., "ZUSD"
        volume = result.get("volume", "0.0")  # Total volume as string

        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        try:
            c = conn.cursor()

            # Create the table if it doesn’t exist
            c.execute("""
                CREATE TABLE IF NOT EXISTS kraken_trading_fees (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER,
                    pair TEXT,
                    fee_type TEXT,
                    fee REAL,
                    minfee REAL,
                    maxfee REAL,
                    nextfee REAL,
                    tiervolume REAL,
                    nextvolume REAL,
                    total_volume REAL,
                    volume_currency TEXT
                )
            """)

            # Insert data with current timestamp
            timestamp = int(time.time())

            # Insert taker fees
            fees = result.get("fees", {})
            for pair, fee_info in fees.items():
                c.execute("""
                    INSERT INTO kraken_trading_fees (
                        timestamp, pair, fee_type, fee, minfee, maxfee, nextfee, tiervolume, nextvolume, total_volume, volume_currency
                    )
                    VALUES (?, ?, 'taker', ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    pair,
                    float(fee_info.get("fee", "0.0")),
                    float(fee_info.get("minfee", "0.0")),
                    float(fee_info.get("maxfee", "0.0")),
                    float(fee_info.get("nextfee")) if fee_info.get("nextfee") is not None else None,
                    float(fee_info.get("tiervolume")) if fee_info.get("tiervolume") is not None else None,
                    float(fee_info.get("nextvolume")) if fee_info.get("nextvolume") is not None else None,
                    float(volume),
                    currency
                ))

            # Insert maker fees
            fees_maker = result.get("fees_maker", {})
            for pair, fee_info in fees_maker.items():
                c.execute("""
                    INSERT INTO kraken_trading_fees (
                        timestamp, pair, fee_type, fee, minfee, maxfee, nextfee, tiervolume, nextvolume, total_volume, volume_currency
                    )
                    VALUES (?, ?, 'maker', ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    timestamp,
                    pair,
                    float(fee_info.get("fee", "0.0")),
                    float(fee_info.get("minfee", "0.0")),
                    float(fee_info.get("maxfee", "0.0")),
                    float(fee_info.get("nextfee")) if fee_info.get("nextfee") is not None else None,
                    float(fee_info.get("tiervolume")) if fee_info.get("tiervolume") is not None else None,
                    float(fee_info.get("nextvolume")) if fee_info.get("nextvolume") is not None else None,
                    float(volume),
                    currency
                ))

            # Commit the transaction
            conn.commit()
            logger.info(f"[KrakenRestManager] Stored trade volume and fees for {len(pair_names)} pairs.")

        except Exception as e:
            logger.exception(f"[KrakenRestManager] Error storing trade volume and fees: {e}")

        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # PUBLIC: GET /0/public/Assets
    # --------------------------------------------------------------------------
    def fetch_asset_info(self, asset_list: List[str], aclass: str = "currency") -> Dict[str, Any]:
        """
        Calls GET /0/public/Assets for the given asset_list, e.g. ["XETH","XXBT"].
        If asset_list is empty => retrieves *all* available assets (careful with size).
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

    # --------------------------------------------------------------------------
    # build_coin_name_lookup_from_db (UPDATED => Single Bulk Fetch or Chunked)
    # --------------------------------------------------------------------------
    def build_coin_name_lookup_from_db(
            self,
            db_path: str = "trades.db",
            lookup_table: str = "kraken_asset_name_lookup",
            chunk_size: int = 50
    ):
        """
        1) Reads 'wsname', 'base', 'pair_name' from 'kraken_asset_pairs' in 'db_path'.
           Only those with a quote='ZUSD' are considered.
        2) Collects unique base assets => e.g. ["XETH","XXBT","XXRP", ...].
        3) Splits them into chunks of up to 'chunk_size' assets each to reduce risk
           of querystring length or timeouts.
        4) For each chunk => calls self.fetch_asset_info(chunk_list) => parse altname, decimals, etc.
        5) Inserts/updates each row in 'kraken_asset_name_lookup'.

        This approach avoids calling fetch_asset_info individually for each base,
        preventing large numbers of small requests that can time out.
        """
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

            # 2) Read relevant rows from kraken_asset_pairs
            z_usd = "ZUSD"
            c.execute(f"""
                SELECT wsname, base, pair_name, altname as alternative_name
                FROM kraken_asset_pairs
                WHERE base IS NOT NULL
                AND quote = '{z_usd}'
            """)
            pair_data = c.fetchall()

            # Build a dict => { base_asset: [(wsname, pair_name, alt_in_db), ...] }
            base_dict = {}
            for (wsname, base_val, pair_name, alt_in_db) in pair_data:
                if not base_val:
                    continue
                if base_val not in base_dict:
                    base_dict[base_val] = []
                base_dict[base_val].append((wsname, pair_name, alt_in_db))

            unique_bases = list(base_dict.keys())
            logger.info(f"[KrakenRestManager] Found {len(unique_bases)} unique base assets with quote=ZUSD.")

            # 3) Split unique_bases into chunks of up to chunk_size
            def chunked(lst, size):
                for i in range(0, len(lst), size):
                    yield lst[i:i + size]

            # For each chunk => call fetch_asset_info once => parse
            for chunk in chunked(unique_bases, chunk_size):
                # 4) fetch them in a single request
                asset_resp = self.fetch_asset_info(chunk, aclass="currency")
                if not asset_resp or "result" not in asset_resp:
                    logger.warning(f"[KrakenRestManager] No 'result' for chunk => skipping.")
                    continue
                result_dict = asset_resp["result"]

                # For each base in this chunk => parse the asset_info fields
                for base_asset in chunk:
                    asset_info_dict = result_dict.get(base_asset, {})
                    if not asset_info_dict:
                        logger.warning(f"[KrakenRestManager] No info for base_asset={base_asset} in chunk => skip.")
                        continue

                    aclass = asset_info_dict.get("aclass", "")
                    altname = asset_info_dict.get("altname", "")
                    decimals = asset_info_dict.get("decimals", 0)
                    disp_decimals = asset_info_dict.get("display_decimals", 0)
                    collateral_val = asset_info_dict.get("collateral_value", None)
                    status_val = asset_info_dict.get("status", "")

                    # Insert/replace each row that references this base_asset
                    row_list = base_dict[base_asset]  # e.g. [(ws, pair_name, alt_in_db), ...]
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
                                altname,
                                aclass,
                                decimals,
                                disp_decimals,
                                float(collateral_val) if collateral_val else None,
                                status_val
                            ))
                        except Exception as e:
                            logger.exception(f"[KrakenRestManager] DB insert error => {e}")

            conn.commit()
            logger.info(f"[KrakenRestManager] build_coin_name_lookup_from_db => updated {lookup_table} table.")
        except Exception as e:
            logger.exception(f"[KrakenRestManager] Error building coin lookup => {e}")
        finally:
            conn.close()

    # --------------------------------------------------------------------------
    # PUBLIC -> PRIVATE -> signing
    # --------------------------------------------------------------------------
    def _public_request(self, url: str, params: dict = None, timeout: int = 30) -> Dict[str, Any]:
        """
        Simple GET for public endpoints. Returns the JSON as a dict, logs if error.
        NOTE: We default to 'timeout=30' to reduce risk of large queries timing out.
        """
        if not params:
            params = {}
        try:
            r = requests.get(url, params=params, timeout=timeout)
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

    def _private_request(self, endpoint: str, payload: dict) -> Dict[str, Any]:
        """
        Creates a nonce, encodes payload, and POSTs with your API-Key, API-Sign headers.
        Returns the JSON response as a dict, or {} if error.
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
            resp = requests.post(url, headers=headers, data=postdata, timeout=30)
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
    # The rest of your private/public endpoints remain the same...
    # (fetch_balance, fetch_and_store_ledger, get_websocket_token, etc.)
    # --------------------------------------------------------------------------

    def fetch_balance(self) -> Dict[str, float]:
        """
        Calls /0/private/Balance => returns a dict {asset: float_balance},
        and also stores them into the DB via store_kraken_balances(...).
        Ensures no scientific notation in logs or storage by formatting floats.
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
                # Convert to float first
                float_val = float(v)
                # Format as a plain decimal string with up to 20 decimal places, stripping trailing zeros
                formatted_val = "{:.20f}".format(float_val).rstrip("0").rstrip(".")
                # Store as float in the output dict (avoids scientific notation internally)
                out[k] = float(formatted_val)
            except (ValueError, TypeError):
                out[k] = 0.0

        # Log with formatted values to avoid scientific notation
        formatted_out = {k: "{:.20f}".format(v).rstrip("0").rstrip(".") for k, v in out.items()}
        logger.info(f"[KrakenRestManager] fetch_balance => {formatted_out}")

        from db import store_kraken_balances
        store_kraken_balances(out)
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
        Calls /0/private/GetWebSocketsToken => returns JSON containing the 'token'.
        """
        endpoint = "/0/private/GetWebSocketsToken"
        resp = self._private_request(endpoint, {})
        if not resp or "result" not in resp:
            logger.warning("[KrakenRestManager] get_websocket_token => no result => skip.")
            return None
        return resp


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    manager = KrakenRestManager(api_key=os.getenv("KRAKEN_API_KEY", ""),
                                api_secret=os.getenv("KRAKEN_SECRET_API_KEY", ""))

    # Example usage:
    print("Balance:", manager.fetch_balance())
    manager.fetch_and_store_ledger(db_path="trades.db")

    # Single fetch of all asset pairs
    pairs_json = manager.fetch_public_asset_pairs()
    if pairs_json:
        # Possibly store them in DB
        print(f"Got {len(pairs_json.get('result', {}))} asset pairs from Kraken's public endpoint.")

    # Bulk build coin name lookup
    manager.build_coin_name_lookup_from_db(db_path="trades.db", chunk_size=50)

    # get WS token
    t = manager.get_websocket_token()
    print("WS token:", t)
