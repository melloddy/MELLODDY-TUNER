# -*- coding: utf-8 -*-

import json
import os.path


class ConfigDict(object):

    def __init__(self, config_file: str = ''):
        """

        :param config_file:
        """
        if os.path.isfile(config_file) is True:
            self._path = 'default' if config_file is None else config_file
            self._params = self._load_param()


    def get_parameters(self, path: str = None):
        """

        :param path:
        :return:
        """
        self._path = self._path if path is None else path
        return self._load_param()

    def _load_param(self):
        """

        :return:
        """
        tmp_dict = {}
        try:
            with open(self._path) as cnf_f:
                tmp_dict = json.load(cnf_f)
        except Exception as e:
            print(e)
        return tmp_dict


parameters = ConfigDict()
