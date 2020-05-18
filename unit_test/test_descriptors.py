 
 
import os
import unittest
import sys
#sys.path.insert(0, '../melloddy_tuner')
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')
import melloddy_tuner as tuner
import melloddy_tuner.chem_utils as chem_utils
import filecmp
import tempfile
from io import BytesIO
import hashlib
import threading
import time
import numpy as np
import pandas as pd

curDir=os.path.dirname(os.path.abspath(__file__))
print(curDir)
 
class DescriptorCalculationTests(unittest.TestCase):
    referenceFilePath=curDir+"/output/smiles_prepared.csv"
    referenceFilePathDuplicates=curDir+"/input/smiles_with_duplicates.csv"

    ############################
    #### setup and teardown ####
    ############################
 
    # executed after each test
    def tearDown(self):
        pass
 
    def defineConfig(self,fp=3):
        if(fp==3):
            tuner.config.parameters.get_parameters(path=curDir+"/../tests/structure_preparation_test/example_parameters.json")
        else:
            tuner.config.parameters.get_parameters(path=curDir+"/input/ecfp2_param.json")

    def defineConfigNewSecret(self):
        tuner.config.parameters.get_parameters(path=curDir+"/input/new_secret_param.json")
###############
#### tests ####
###############
 

    def test_calculate_desc_feat_single(self):
        self.defineConfig()
        tempFilePath=curDir+"/output/tmp/ecfp_feat.npy"

        ecfp=chem_utils.run_fingerprint(["Cc1ccc(S(=O)(=O)Nc2ccc(-c3nc4cc(NS(=O)(=O)c5ccc(C)cc5)ccc4[nH]3)cc2)cc1"],1)
        ecfp_feat, ecfp_val=chem_utils.make_desc_dict(ecfp)
        #np.save("unit_test/output/test_calculate_desc_feat.npy",ecfp_feat)   #write reference fingperprints
        np.save(tempFilePath,ecfp_feat)   #write reference fingperprints

        result=filecmp.cmp("unit_test/output/test_calculate_desc_feat.npy",tempFilePath, shallow=False)

        self.assertEqual(result, True)


    def test_calculate_desc_val_single(self):
        self.defineConfig()
        tempFilePath=curDir+"/output/tmp/ecfp_val.npy"

        ecfp=chem_utils.run_fingerprint(["Cc1ccc(S(=O)(=O)Nc2ccc(-c3nc4cc(NS(=O)(=O)c5ccc(C)cc5)ccc4[nH]3)cc2)cc1"],1)
        ecfp_feat, ecfp_val=chem_utils.make_desc_dict(ecfp)
        #np.save("unit_test/output/test_calculate_desc_val.npy",ecfp_val)   #write reference fingperprints
        np.save(tempFilePath,ecfp_val)   #write reference fingperprints

        result=filecmp.cmp("unit_test/output/test_calculate_desc_val.npy",tempFilePath, shallow=False)

        self.assertEqual(result, True)

    def test_calculate_desc_feat_multiple(self):
        self.defineConfig()
        tempFilePath=curDir+"/output/tmp/ecfp_feat_multiple.npy"
        with open(self.referenceFilePath,"r") as h:
            smiles=[line.strip() for line in h.readlines()]
            ecfps=chem_utils.run_fingerprint(smiles,1)
            desc=chem_utils.make_desc_dict(ecfps)
            #np.save("unit_test/output/test_calculate_desc.npy",desc)   #write reference fingperprints
            np.save(tempFilePath,desc)   #write reference fingperprints
            result=filecmp.cmp("unit_test/output/test_calculate_desc.npy",tempFilePath, shallow=False)
            self.assertEqual(result, True)

    def test_scramble_desc_multiple(self):
        """test if scrambling input fp's is working compared to a reference"""
        self.defineConfig()
        tempFilePathFeat=curDir+"/output/tmp/ecfp_feat_scrambled.npy"
        tempFilePathVal=curDir+"/output/tmp/ecfp_val_scrambled.npy"

        with open(self.referenceFilePath,"r") as h:
            smiles=[line.strip() for line in h.readlines()]
            ecfps=chem_utils.run_fingerprint(smiles,1)
            ecfp_feat, ecfp_val = chem_utils.make_desc_dict(ecfps)
            ecfp_feat_scrambled,ecfp_val_scrambled=chem_utils.make_fp_lists(ecfp_feat,ecfp_val)
            #np.save("unit_test/output/test_calculate_desc_feat_scr.npy",ecfp_feat_scrambled)   #write reference fingperprints
            #np.save("unit_test/output/test_calculate_desc_val_scr.npy",ecfp_val_scrambled)   #write reference fingperprints
            np.save(tempFilePathFeat,ecfp_feat_scrambled)   
            np.save(tempFilePathVal,ecfp_val_scrambled)
            resultFeat=filecmp.cmp("unit_test/output/test_calculate_desc_feat_scr.npy",tempFilePathFeat, shallow=False)
            resultVal=filecmp.cmp("unit_test/output/test_calculate_desc_val_scr.npy",tempFilePathVal, shallow=False)
            self.assertEqual(resultFeat & resultVal, True)

    def test_scramble_desc_multiple_key(self):
        """test if scrambling is depending on the input key"""
        self.defineConfigNewSecret()
        tempFilePathFeat=curDir+"/output/tmp/ecfp_feat_scrambled.npy"
        tempFilePathVal=curDir+"/output/tmp/ecfp_val_scrambled.npy"

        with open(self.referenceFilePath,"r") as h:
            smiles=[line.strip() for line in h.readlines()]
            ecfps=chem_utils.run_fingerprint(smiles,1)
            ecfp_feat, ecfp_val = chem_utils.make_desc_dict(ecfps)
            ecfp_feat_scrambled,ecfp_val_scrambled=chem_utils.make_fp_lists(ecfp_feat,ecfp_val)
            np.save(tempFilePathFeat,ecfp_feat_scrambled)   
            np.save(tempFilePathVal,ecfp_val_scrambled)
            resultFeat=filecmp.cmp("unit_test/output/test_calculate_desc_feat_scr.npy",tempFilePathFeat, shallow=False)
            resultVal=filecmp.cmp("unit_test/output/test_calculate_desc_val_scr.npy",tempFilePathVal, shallow=False)
            self.assertEqual(resultFeat & resultVal, False)

    def test_output_descriptor_duplicates(self):
        """test output for descriptor duplicates"""
        self.defineConfig()
        structure_data = tuner.helper.read_csv("tests/structure_preparation_test/reference_set.csv")
        ecfp = tuner.run_fingerprint(structure_data['smiles'], 1)
        df_processed_desc = tuner.output_processed_descriptors(ecfp, structure_data)
        structure_data_duplicates = tuner.output_descriptor_duplicates(df_processed_desc)
        self.assertEqual(len(structure_data_duplicates), 0)
    
    def test_output_descriptor_duplicates_ref_file(self):
        """test output for descriptor duplicates"""
        self.defineConfig()
        with open(self.referenceFilePathDuplicates,"r") as h:
            smiles=[line.strip() for line in h.readlines()]
            structure_data=pd.DataFrame(smiles,columns=["smiles"])
        
            ecfp = tuner.run_fingerprint(structure_data['smiles'], 1)
            df_processed_desc = tuner.output_processed_descriptors(ecfp, structure_data)
            structure_data_duplicates = tuner.output_descriptor_duplicates(df_processed_desc)
            self.assertEqual(len(structure_data_duplicates), 7)

    def test_output_descriptor_duplicates_ref_file_ecfp1(self):
        """test output for descriptor duplicates with fuzzier fingerprint """
        self.defineConfig(fp=1)
        with open(self.referenceFilePathDuplicates,"r") as h:
            smiles=[line.strip() for line in h.readlines()]
            structure_data=pd.DataFrame(smiles,columns=["smiles"])
        
            ecfp = tuner.run_fingerprint(structure_data['smiles'], 1)
            df_processed_desc = tuner.output_processed_descriptors(ecfp, structure_data)
            structure_data_duplicates = tuner.output_descriptor_duplicates(df_processed_desc)
            self.assertEqual(len(structure_data_duplicates), 19)


    def test_lsh_folding(self):
        self.defineConfig()
        with open(self.referenceFilePathDuplicates,"r") as h:
            smiles=[line.strip() for line in h.readlines()]
            structure_data=pd.DataFrame(smiles,columns=["smiles"])
        
            ecfp = tuner.run_fingerprint(structure_data['smiles'], 1)
            lsh_folding = tuner.LSHFolding()
            df_high_entropy_bits = lsh_folding.calc_highest_entropy_bits(ecfp)
            #df_high_entropy_bits.to_pickle("unit_test/output/df_high_entropy_bits.pkl") #reference results
            df_high_entropy_bits_ref=pd.read_pickle("unit_test/output/df_high_entropy_bits.pkl")

            df_folds = lsh_folding.run_lsh_calculation(ecfp)
            #df_folds.to_pickle("unit_test/output/df_folds.pkl") #reference results
            df_folds_ref = pd.read_pickle("unit_test/output/df_folds.pkl")

            self.assertEqual(df_high_entropy_bits.equals(df_high_entropy_bits_ref) & df_folds.equals(df_folds_ref),True)
            

    def test_lsh_folding_ecfp1(self):
        """test if folding is dependent on fingerprint"""
        self.defineConfig(fp=1)
        with open(self.referenceFilePathDuplicates,"r") as h:
            smiles=[line.strip() for line in h.readlines()]
            structure_data=pd.DataFrame(smiles,columns=["smiles"])
        
            ecfp = tuner.run_fingerprint(structure_data['smiles'], 1)
            lsh_folding = tuner.LSHFolding()
            df_high_entropy_bits = lsh_folding.calc_highest_entropy_bits(ecfp)
            #df_high_entropy_bits.to_pickle("unit_test/output/df_high_entropy_bits.pkl") #reference results
            df_high_entropy_bits_ref=pd.read_pickle("unit_test/output/df_high_entropy_bits.pkl")

            df_folds = lsh_folding.run_lsh_calculation(ecfp)
            #df_folds.to_pickle("unit_test/output/df_folds.pkl") #reference results
            df_folds_ref = pd.read_pickle("unit_test/output/df_folds.pkl")

            self.assertEqual(df_high_entropy_bits.equals(df_high_entropy_bits_ref) & df_folds.equals(df_folds_ref),False)
            
    def test_lsh_folding_new_key(self):
        """test if folding is dependent on key"""
        self.defineConfigNewSecret()
        with open(self.referenceFilePathDuplicates,"r") as h:
            smiles=[line.strip() for line in h.readlines()]
            structure_data=pd.DataFrame(smiles,columns=["smiles"])
        
            ecfp = tuner.run_fingerprint(structure_data['smiles'], 1)
            lsh_folding = tuner.LSHFolding()
            df_high_entropy_bits = lsh_folding.calc_highest_entropy_bits(ecfp)
            #df_high_entropy_bits.to_pickle("unit_test/output/df_high_entropy_bits.pkl") #reference results
            df_high_entropy_bits_ref=pd.read_pickle("unit_test/output/df_high_entropy_bits.pkl")

            df_folds = lsh_folding.run_lsh_calculation(ecfp)
            #df_folds.to_pickle("unit_test/output/df_folds.pkl") #reference results
            df_folds_ref = pd.read_pickle("unit_test/output/df_folds.pkl")

            self.assertEqual(df_high_entropy_bits.equals(df_high_entropy_bits_ref) & df_folds.equals(df_folds_ref),False)
            


#if __name__ == "__main__":
#    unittest.main()
