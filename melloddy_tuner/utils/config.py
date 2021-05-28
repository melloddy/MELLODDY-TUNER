# -*- coding: utf-8 -*-

import json
import os.path
from pathlib import Path
import typing


class ConfigDict(object):
    """
    Config Class to read config file and load config parameters.



    """

    _CONFIG_FILE: typing.Optional[str] = ""
    _CONFIG: typing.Optional[dict] = {}

    def __init__(self, config_path: Path = None) -> None:
        """
        Initialize config class by reading config from file.

        Args:
            config_path (Path, optional): path to config file. Defaults to None.
        """
        if config_path is not None:
            ConfigDict._CONFIG_FILE = config_path
            self._config_path = config_path

            ConfigDict._CONFIG = self._load_cfg()

    @staticmethod
    def get_config_file() -> str:
        return ConfigDict._CONFIG_FILE

    @staticmethod
    def get_parameters() -> dict:
        return ConfigDict._CONFIG

    def _load_cfg(self):

        tmp_dict = {}
        # if self._config_path.suffix == ".json":
        try:
            with open(self._config_path) as cnf_f:
                tmp_dict = json.load(cnf_f)
        except Exception as e:
            print(e)
        return tmp_dict
        # else:
        #     print("Not allowed config format. Please use .json file.")
        #     return quit()


class SecretDict(object):
    """
    Secret Class to read key file and load secret key.
    """

    _KEY_FILE: typing.Optional[str] = ""
    _KEY: typing.Optional[dict] = {}

    def __init__(self, key_path=None):
        """

        :param key_path:
        """
        if key_path is not None:
            SecretDict._KEY_FILE = key_path
            self._key_path = key_path
            SecretDict._KEY = self._load_key()

    @staticmethod
    def get_key_file() -> str:
        return SecretDict._KEY_FILE

    @staticmethod
    def get_secrets() -> dict:
        return SecretDict._KEY

    def _load_key(self):
        """

        :return:
        """
        tmp_dict = {}
        # if self._key_path.suffix == ".json":
        try:
            with open(self._key_path) as key_f:
                tmp_dict = json.load(key_f)
        except Exception as e:
            print(e)
        return tmp_dict
        # else:
        #     print("Not allowed config format. Please use .json file.")
        #     return quit()


parameters = ConfigDict()
secrets = SecretDict()
