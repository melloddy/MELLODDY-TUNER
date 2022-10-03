import os
import unittest
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import json
from melloddy_tuner.utils.single_row_prep2pred import SingleRowPreparator
from melloddy_tuner.utils.config import ConfigDict, SecretDict, parameters, secrets

curDir = Path(os.path.dirname(os.path.abspath(__file__)))
print(curDir)

class Prepare2PredTests(unittest.TestCase):
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

    def test_single_to_coo(self):
        my_srprep = SingleRowPreparator(params = self.config, secret = self.keys)	
        #read in similes an create coo tensors from them
        df_smiles = pd.read_csv((curDir / "input/chembl/chembl_23_example_T2.csv"), nrows=10,index_col="input_compound_id")
        test_tensors = {i : my_srprep.process_smiles(df_smiles.loc[i,"smiles"]) for i in df_smiles.index}
		
        #read in the corrsponding fingerprint features and values calculated from csv in the conventional way and convert into torch tensors
		#this test doesn't use a save coo tensor file directly, since torch save format is about to change in pytorch 1.6
        df_fp = pd.read_csv((curDir / "output/test_calculate_desc_y2.csv"), nrows=10,index_col="input_compound_id")
        ref_tensors = {}
        fp_size = self.config['fingerprint']['fold_size']
        for i in df_fp.index:
            features = np.array(json.loads(df_fp.loc[i,'fp_feat']))
            values = np.array(json.loads(df_fp.loc[i,'fp_val']))
            row_ind = np.repeat(0, features.shape[0])
            ref_tensors[i] = torch.sparse_coo_tensor(indices = np.array([row_ind,features]), values = values, size = (1, fp_size),dtype=torch.float).coalesce()
        #compare the tensors with allclose			
        test_res = {id : torch.allclose(test_tensor.to_dense(), ref_tensors[id].to_dense()) for id, test_tensor in test_tensors.items()}
        self.assertEqual(all(test_res.values()), True)		