"""
config_loader.py

A revised, highly-versatile module that provides convenient methods to load and access
configuration data from a YAML file. This version reloads the config from disk on every
getter call, ensuring that any external updates to the file are automatically picked up.
Setters also save changes back to the file immediately, preserving any updated values.

Typical usage:
    from config_loader import ConfigLoader

    # Optionally specify an alternate config file path (default is "config.yaml"):
    ConfigLoader.initialize("my_config.yaml")

    # Get a config value:
    value = ConfigLoader.get_value("some_nested.key.path")

    # Get a list of traded pairs:
    pairs = ConfigLoader.get_traded_pairs()

    # Set a value:
    ConfigLoader.set_value("risk_controls.max_position_value", 30.0)
    # This automatically saves the update to the YAML file.
"""

import os
import yaml
from typing import Any, List


class ConfigLoader:
    """
    A class that abstracts the process of loading, reading, and writing YAML configuration data.
    - Automatically re-reads the config file on every get_* call, ensuring the freshest config.
    - Automatically saves updated config values to file on every set_* call.

    This approach is useful when you want real-time changes to your config without restarting
    your application. However, re-loading the file on every access can be expensive in high-frequency
    use cases, so keep performance considerations in mind.
    """

    _config_file_path: str = "config.yaml"
    _config_data: dict = {}

    @classmethod
    def initialize(cls, config_file_path: str = "config.yaml") -> None:
        """
        Optionally initializes the loader with a specific config file path.
        If not called, the loader will default to 'config.yaml'.

        :param config_file_path: The path to the YAML config file.
        :type config_file_path: str
        """
        cls._config_file_path = config_file_path

    @classmethod
    def _load_config(cls) -> None:
        """
        Loads the YAML configuration from the current _config_file_path into _config_data.

        :raises FileNotFoundError: If the specified config file does not exist.
        :raises yaml.YAMLError: If the YAML file has a parsing error.
        """
        if not os.path.isfile(cls._config_file_path):
            raise FileNotFoundError(f"Configuration file not found at: {cls._config_file_path}")

        try:
            with open(cls._config_file_path, "r", encoding="utf-8") as file:
                cls._config_data = yaml.safe_load(file) or {}
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML configuration: {e}")

    @classmethod
    def _save_config(cls) -> None:
        """
        Saves the current in-memory configuration data (_config_data) back to the file specified
        by _config_file_path.

        :raises RuntimeError: If an error occurs while writing the file.
        """
        try:
            with open(cls._config_file_path, "w", encoding="utf-8") as file:
                yaml.safe_dump(cls._config_data, file)
        except Exception as e:
            raise RuntimeError(f"Failed to save configuration to {cls._config_file_path}: {e}")

    @classmethod
    def _reload_config(cls) -> None:
        """
        A helper method that reloads the config from disk. This is called automatically
        on getter methods to ensure the freshest config if the file has changed.
        """
        cls._load_config()

    @classmethod
    def get_value(cls, key: str, default: Any = None) -> Any:
        """
        Retrieves a configuration value by key (including nested paths, e.g. "risk_controls.minimum_buy_amount").
        Automatically reloads the config file each time to get the latest external changes.

        :param key: The top-level key or a dotted path to a nested key.
        :type key: str
        :param default: A default value to return if the key path does not exist.
        :type default: Any
        :return: The value from the config if it exists, otherwise returns default.
        :rtype: Any

        :example:
            minimum_buy = ConfigLoader.get_value("risk_controls.minimum_buy_amount", default=0.0)
        """
        cls._reload_config()

        # Traverse nested keys if present
        keys = key.split(".")
        data = cls._config_data
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return default
        return data

    @classmethod
    def get_traded_pairs(cls) -> List[str]:
        """
        Retrieves the latest list of traded pairs from the configuration file.

        :return: A list of traded pair symbols, e.g. ["ETH/USD", "XBT/USD", "SOL/USD", ...].
        :rtype: List[str]
        """
        cls._reload_config()

        traded_pairs = cls._config_data.get("traded_pairs", [])
        return traded_pairs if isinstance(traded_pairs, list) else []

    @classmethod
    def set_value(cls, key: str, value: Any) -> None:
        """
        Dynamically update a configuration value and save the change back to the file.
        This ensures that subsequent reads (and other processes watching the config)
        reflect the updated value immediately.

        :param key: The key (or nested key path) to update, e.g. "risk_controls.purchase_upper_limit_percent".
        :type key: str
        :param value: The new value to assign to that key path.
        :type value: Any

        :example:
            ConfigLoader.set_value("risk_controls.purchase_upper_limit_percent", 80.0)
        """
        # Reload the config to ensure we have the latest data
        cls._reload_config()

        # Traverse or create nested dictionaries
        keys = key.split(".")
        data = cls._config_data
        for k in keys[:-1]:
            if k not in data or not isinstance(data[k], dict):
                data[k] = {}
            data = data[k]
        data[keys[-1]] = value

        # Now save config to file so that this update is persisted
        cls._save_config()

    @classmethod
    def get_all_keys(cls) -> List[str]:
        """
        Reloads the config, then returns a list of top-level keys in the configuration.

        :return: A list of top-level keys.
        :rtype: List[str]
        """
        cls._reload_config()
        return list(cls._config_data.keys())
