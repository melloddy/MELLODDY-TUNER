import os
import unittest
import sys

# sys.path.insert(0, '../melloddy_tuner')
from pathlib import Path
from rdkit import rdBase

rdBase.DisableLog("rdApp.*")

from melloddy_tuner.utils.formatting import ActivityDataFormatting
from melloddy_tuner.utils.config import parameters, secrets
from melloddy_tuner.utils.helper import read_csv
import filecmp
import tempfile
from io import BytesIO
import hashlib
import threading
import time
import numpy as np
import pandas as pd

curDir = Path(os.path.dirname(os.path.abspath(__file__)))


class ActivityFormattingTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(ActivityFormattingTests, self).__init__(*args, **kwargs)
        self.referenceFilePath = curDir / "output/smiles_prepared.csv"
        self.referenceFilePathDuplicates = curDir / "input/smiles_with_duplicates.csv"
        self.path_mapping_table_T5 = (
            curDir / "output/chembl/results_tmp/mapping_table_T5.csv"
        )
        self.path_mapping_table_T10 = (
            curDir / "output/chembl/results_tmp/mapping_table_T10.csv"
        )
        self.path_T2_T6 = curDir / "output/chembl/results_tmp/T6.csv"
        self.activity_file = curDir / "input/chembl/chembl_23_example_T4.csv"
        self.defineConfig()

        mapping_table_T5 = read_csv(self.path_mapping_table_T5)
        activity_data = read_csv(self.activity_file)
        mapping_table_T10 = read_csv(self.path_mapping_table_T10)
        self.act_data_format = ActivityDataFormatting(
            activity_data, mapping_table_T5, mapping_table_T10
        )
        del (activity_data, mapping_table_T5, mapping_table_T10)
        self.act_data_format.run_formatting()

    ############################
    #### setup and teardown ####
    ############################

    # executed after each test
    def tearDown(self):
        pass

    def defineConfig(self, fp=3):
        if fp == 3:

            parameters.__init__(
                config_path=curDir / "reference_files/example_parameters.json"
            )
        else:
            parameters.__init__(config_path=curDir / "input/ecfp2_param.json")

    def defineKey(self):
        secrets.__init__(key_path=curDir / "reference_files/example_key.json")

    def defineConfigNewSecret(self):
        secrets.__init__(key_path=curDir / "input/new_secret_param.json")

    ###############
    #### tests ####
    ###############

    def test_activity_formatting_chembl_failed(self):
        # tempFilePath=curDir / "output/tmp/activity_failed.npy"
        data_failed = self.act_data_format.filter_failed_structures()
        # data_failed.to_pickle(curDir / "output/test_activity_formatting_failed.pkl")   #save referene data

        data_failed_ref = pd.read_pickle(
            curDir / "output/test_activity_formatting_failed.pkl"
        )

        self.assertEqual(data_failed.equals(data_failed_ref), True)

    def test_activity_formatting_duplicates(self):
        data_duplicates = self.act_data_format.data_duplicates
        # data_duplicates.to_pickle(curDir / "output/test_activity_formatting_duplicates.pkl")   #save referene data
        data_duplicates_ref = pd.read_pickle(
            curDir / "output/test_activity_formatting_duplicates.pkl"
        )
        self.assertEqual(data_duplicates.equals(data_duplicates_ref), True)

    def test_activity_formatting_excluded(self):
        data_excluded = self.act_data_format.select_excluded_data()
        # data_excluded.to_pickle(curDir / "output/test_activity_formatting_excluded.pkl")   #save referene data
        data_excluded_ref = pd.read_pickle(
            curDir / "output/test_activity_formatting_excluded.pkl"
        )
        self.assertEqual(data_excluded.equals(data_excluded_ref), True)

    def test_activity_formatting_chembl_t11(self):
        self.act_data_format.remapping_2_cont_ids()
        structure_data_T6 = read_csv(self.path_T2_T6)
        structure_data_T11 = self.act_data_format.make_T11(
            structure_data_T6
        ).sort_values("cont_descriptor_vector_id")
        #         structure_data_T11.to_pickle(curDir / "output/test_activity_formatting_t11.pkl")   #save referene data
        structure_data_T11_ref = pd.read_pickle(
            curDir / "output/test_activity_formatting_t11.pkl"
        )
        self.assertEqual(structure_data_T11.equals(structure_data_T11_ref), True)

    def test_activity_formatting_chembl_t10(self):
        self.act_data_format.remapping_2_cont_ids()
        data_remapped = self.act_data_format.data_remapped.sort_values(
            "cont_classification_task_id"
        )
        #         data_remapped.to_pickle(curDir / "output/test_activity_formatting_t10.pkl")   #save referene data
        structure_data_T10_ref = pd.read_pickle(
            curDir / "output/test_activity_formatting_t10.pkl"
        )
        self.assertEqual(data_remapped.equals(structure_data_T10_ref), True)

        # print(data_failed)
        # print(act_data_format)


# if __name__ == "__main__":
#    unittest.main()
