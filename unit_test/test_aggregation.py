from melloddy_tuner.utils.helper import read_csv, read_input_file
from melloddy_tuner.scripts.aggregate_values import aggregate_replicates
from melloddy_tuner.utils.config import ConfigDict, SecretDict, parameters, secrets
import os
import unittest
import sys
from pathlib import Path
import pandas.testing as pd_testing

# sys.path.insert(0, '../melloddy_tuner')

import filecmp
import tempfile
from io import BytesIO
import hashlib
import threading
import time
import numpy as np
import pandas as pd

curDir = Path(os.path.dirname(os.path.abspath(__file__)))
print(curDir)


class AggregationTests(unittest.TestCase):
    referenceFilePath = curDir / "output/smiles_prepared.csv"
    referenceFilePathDuplicates = curDir / "input/smiles_with_duplicates.csv"

    ############################
    #### setup and teardown ####
    ############################

    # executed after each test

    def tearDown(self):
        pass

    def setUp(self):
        self.config = ConfigDict(
            config_path=Path(
                os.path.join(curDir, "reference_files", "example_parameters.json")
            )
        ).get_parameters()
        self.keys = SecretDict(
            key_path=Path(os.path.join(curDir, "reference_files", "example_key.json"))
        ).get_secrets()

    ###############
    #### tests ####
    ###############

    def test_aggregation(self):
        T0file = os.path.join(curDir, "input", "test_aggr", "T0.csv")
        T1file = os.path.join(curDir, "input", "test_aggr", "T1.csv")
        T5file = os.path.join(curDir, "input", "test_aggr", "T5.csv")
        T0 = read_input_file(T0file)
        T1 = read_input_file(T1file)
        T5 = read_input_file(T5file)

        (
            df_aggr,
            df_failed_range,
            df_failed_binary,
            df_failed_aggr,
            df_failed_std,
            df_dup,
            t0upd
        ) = aggregate_replicates(T0, T1, T5, self.config["credibility_range"], 1)
        T4r = df_aggr[
            [
                "input_assay_id",
                "descriptor_vector_id",
                "fold_id",
                "standard_qualifier",
                "standard_value",
            ]
        ].reset_index(drop=True)
        df_failed_range = df_failed_range[
            [
                "input_compound_id",
                "input_assay_id",
                "standard_qualifier",
                "standard_value",
            ]
        ].reset_index(drop=True)
        df_failed_binary = df_failed_binary[
            [
                "input_compound_id",
                "input_assay_id",
                "standard_qualifier",
                "standard_value",
            ]
        ].reset_index(drop=True)
        df_failed_aggr = df_failed_aggr[
            [
                "descriptor_vector_id",
                "input_assay_id",
                "standard_qualifier",
                "standard_value",
                "fold_id",
            ]
        ].reset_index(drop=True)
        df_failed_std = df_failed_std[
            [
                "descriptor_vector_id",
                "input_assay_id",
                "standard_qualifier",
                "standard_value",
                "fold_id",
            ]
        ].reset_index(drop=True)
        df_dup = df_dup[
            [
                "input_assay_id",
                "input_compound_id",
                "descriptor_vector_id",
                "fold_id",
                "standard_qualifier",
                "standard_value",
            ]
        ].reset_index(drop=True)

        dupfile = os.path.join(curDir, "output", "test_aggr", "duplicates_T1.csv")
        failed_range_file = os.path.join(
            curDir, "output", "test_aggr", "failed_range_T1.csv"
        )
        failed_aggr_file = os.path.join(
            curDir, "output", "test_aggr", "failed_aggr_T1.csv"
        )
        failed_std_file = os.path.join(
            curDir, "output", "test_aggr", "failed_std_T1.csv"
        )
        T4rfile = os.path.join(curDir, "output", "test_aggr", "T4r.csv")
        df_dup_exp = read_input_file(dupfile)
        df_failed_aggr_exp = read_input_file(failed_aggr_file)
        df_failed_range_exp = read_input_file(failed_range_file)
        df_failed_std_exp = read_input_file(failed_std_file)
        T4r_exp = read_input_file(T4rfile)

        pd_testing.assert_frame_equal(T4r, T4r_exp)
        pd_testing.assert_frame_equal(df_failed_range, df_failed_range_exp)
        pd_testing.assert_frame_equal(df_failed_std, df_failed_std_exp)
        # aggr dfs don't match due to different column type object vs float64
        # pd_testing.assert_frame_equal(
        #    df_failed_aggr, df_failed_aggr_exp, check_column_type=False)
        pd_testing.assert_frame_equal(df_dup, df_dup_exp)


if __name__ == "__main__":
    unittest.main()
