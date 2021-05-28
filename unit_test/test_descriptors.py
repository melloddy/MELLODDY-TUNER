from melloddy_tuner.utils.helper import format_dataframe
from melloddy_tuner.utils.df_transformer import DfTransformer
from melloddy_tuner.utils.helper import read_csv
from melloddy_tuner.utils.descriptor_calculator import DescriptorCalculator
from melloddy_tuner.utils.chem_utils import run_fingerprint
from melloddy_tuner.utils.folding import LSHFolding
from melloddy_tuner.utils.config import ConfigDict, SecretDict, parameters, secrets
import os
import unittest
import sys
from pathlib import Path

# sys.path.insert(0, '../melloddy_tuner')
from rdkit import rdBase

rdBase.DisableLog("rdApp.*")
from melloddy_tuner import utils
from melloddy_tuner.utils import chem_utils

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


class DescriptorCalculationTests(unittest.TestCase):
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

    # def defineConfig(self,fp=3):
    #     if(fp==3):
    #         config_file = curDir / "reference_files/example_parameters.json"
    #         if parameters.get_config_file() != config_file:
    #             parameters.__init__(config_path=config_file)
    #     else:
    #         ecfp_conf = curDir / "input/ecfp2_param.json"
    #         if parameters.get_config_file() != ecfp_conf:
    #             parameters.__init__(config_path=ecfp_conf)

    # def defineKey(self):
    #     key_file = curDir / "reference_files/example_key.json"
    #     if secrets.get_key_file() != key_file:
    #         secrets.__init__(key_path=key_file)

    # def defineNewKey(self):
    #     key_new = curDir / "input/new_secret.json"
    #     if secrets.get_key_file() != key_new:
    #         secrets.__init__(key_path=key_new)

    ###############
    #### tests ####
    ###############

    def test_calculate_desc_feat_single(self):
        # self.defineConfig()
        tempFilePath = curDir / "output/tmp/ecfp_feat.npy"
        dc = DescriptorCalculator.from_param_dict(
            secret=self.keys["key"],
            method_param_dict=self.config["fingerprint"],
            verbosity=0,
        )
        fp = dc.calculate_single(
            "Cc1ccc(S(=O)(=O)Nc2ccc(-c3nc4cc(NS(=O)(=O)c5ccc(C)cc5)ccc4[nH]3)cc2)cc1"
        )
        fp_feat = fp[0]

        # np.save("unit_test/output/test_calculate_desc_feat.npy",fp_feat)   #write reference fingperprints
        np.save(tempFilePath, fp_feat)  # write test fingperprints

        result = filecmp.cmp(
            "unit_test/output/test_calculate_desc_feat.npy", tempFilePath, shallow=False
        )

        self.assertEqual(result, True)

    def test_calculate_desc_val_single(self):
        tempFilePath = curDir / "output/tmp/ecfp_val.npy"
        dc = DescriptorCalculator.from_param_dict(
            secret=self.keys["key"],
            method_param_dict=self.config["fingerprint"],
            verbosity=0,
        )
        fp = dc.calculate_single(
            "Cc1ccc(S(=O)(=O)Nc2ccc(-c3nc4cc(NS(=O)(=O)c5ccc(C)cc5)ccc4[nH]3)cc2)cc1"
        )
        fp_val = fp[1]
        # np.save("unit_test/output/test_calculate_desc_val.npy",fp_val)   #write reference fingperprints
        np.save(tempFilePath, fp_val)  # write test fingperprints

        result = filecmp.cmp(
            "unit_test/output/test_calculate_desc_val.npy", tempFilePath, shallow=False
        )

        self.assertEqual(result, True)

    def test_calculate_desc_multiple(self):
        tempFilePath = curDir / "output/tmp/ecfp_feat_multiple.csv"
        df_smiles = read_csv(curDir / "input/chembl/chembl_23_example_T2.csv", nrows=10)

        dc = DescriptorCalculator.from_param_dict(
            secret=self.keys["key"],
            method_param_dict=self.config["fingerprint"],
            verbosity=0,
        )
        outcols = ["fp_feat", "fp_val", "success", "error_message"]
        out_types = ["object", "object", "bool", "object"]
        dt = DfTransformer(
            dc,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=1,
            verbosity=0,
        )
        # df_ref = dt.process_dataframe(df_smiles)[0] #calculate reference fingperprints
        # df_ref.to_csv("unit_test/output/test_calculate_desc_y2.csv", index=False)   #write reference fingperprints

        df_test = dt.process_dataframe(df_smiles)[0]
        df_test.to_csv(tempFilePath, index=False)  # write test fingperprints
        result = filecmp.cmp(
            "unit_test/output/test_calculate_desc_y2.csv", tempFilePath, shallow=False
        )

        self.assertEqual(result, True)

    def test_scramble_desc_multiple_key(self):
        """test if scrambling is depending on the input key"""
        newKey = "melloddy_2"

        tempFilePathFeat = curDir / "output/tmp/ecfp_feat_scrambled_new_key.csv"
        df_smiles = read_csv(curDir / "input/chembl/chembl_23_example_T2.csv", nrows=10)

        dc = DescriptorCalculator.from_param_dict(
            secret=newKey, method_param_dict=self.config["fingerprint"], verbosity=0
        )
        outcols = ["fp_feat", "fp_val", "success", "error_message"]
        out_types = ["object", "object", "bool", "object"]
        dt = DfTransformer(
            dc,
            input_columns={"smiles": "smiles"},
            output_columns=outcols,
            output_types=out_types,
            success_column="success",
            nproc=1,
            verbosity=0,
        )
        df_test = dt.process_dataframe(df_smiles)[0]
        df_test.to_csv(tempFilePathFeat, index=False)  # write test fingperprints
        result = filecmp.cmp(
            "unit_test/output/test_calculate_desc_y2.csv",
            tempFilePathFeat,
            shallow=False,
        )
        self.assertEqual(result, False)

    def test_output_descriptor_duplicates(self):
        """test output for descriptor duplicates"""
        tempFilePathFeat_dup = curDir / "output/tmp/desc_duplicates.csv"
        tempFilePathFeat_T5 = curDir / "output/tmp/desc_T5.csv"
        tempFilePathFeat_T6 = curDir / "output/tmp/desc_T6.csv"
        df_test = read_csv(curDir / "input/test_desc_duplicates.csv")

        df_T5, df_T6, df_duplicates = format_dataframe(df_test)

        # ensure compleness of input covered in T5
        result = set(df_test["input_compound_id"]) == set(df_T5["input_compound_id"])
        self.assertEqual(result, True)

        # ensure absence of null values in T5
        result = df_T5["descriptor_vector_id"].notnull().all()
        self.assertEqual(result, True)

        # compare with referenc e files
        df_duplicates.to_csv(tempFilePathFeat_dup, index=False)
        df_T6.to_csv(tempFilePathFeat_T6, index=False)
        df_T5.to_csv(tempFilePathFeat_T5, index=False)

        result = filecmp.cmp(
            "unit_test/output/test_calculate_desc_duplicates.csv",
            tempFilePathFeat_dup,
            shallow=False,
        )
        self.assertEqual(result, True)
        result = filecmp.cmp(
            "unit_test/output/test_calculate_desc_T5.csv",
            tempFilePathFeat_T5,
            shallow=False,
        )
        self.assertEqual(result, True)
        result = filecmp.cmp(
            "unit_test/output/test_calculate_desc_T6.csv",
            tempFilePathFeat_T6,
            shallow=False,
        )
        self.assertEqual(result, True)

    # def test_output_descriptor_duplicates_ref_file(self):
    #     """test output for descriptor duplicates"""
    #     self.defineConfig()
    #     self.defineKey()
    #     with open(self.referenceFilePathDuplicates,"r") as h:
    #         smiles=[line.strip() for line in h.readlines()]
    #         structure_data=pd.DataFrame(smiles,columns=["smiles"])

    #         ecfp = run_fingerprint(structure_data['smiles'], 1)
    #         df_processed_desc = chem_utils.output_processed_descriptors(ecfp, structure_data)
    #         structure_data_duplicates = chem_utils.output_descriptor_duplicates(df_processed_desc)
    #         self.assertEqual(len(structure_data_duplicates), 7)

    # def test_output_descriptor_duplicates_ref_file_ecfp1(self):
    #     """test output for descriptor duplicates with fuzzier fingerprint """
    #     self.defineConfig(fp=1)
    #     self.defineKey()
    #     with open(self.referenceFilePathDuplicates,"r") as h:
    #         smiles=[line.strip() for line in h.readlines()]
    #         structure_data=pd.DataFrame(smiles,columns=["smiles"])

    #         ecfp = run_fingerprint(structure_data['smiles'], 1)
    #         df_processed_desc = chem_utils.output_processed_descriptors(ecfp, structure_data)
    #         structure_data_duplicates = chem_utils.output_descriptor_duplicates(df_processed_desc)
    #         self.assertEqual(len(structure_data_duplicates), 19)


# if __name__ == "__main__":
#    unittest.main()
