 
 
import os
import unittest
import sys
#sys.path.insert(0, '../melloddy_tuner')
from rdkit import rdBase
rdBase.DisableLog('rdApp.*')
import melloddy_tuner.chem_utils as chem_utils
import melloddy_tuner as tuner
import filecmp
import tempfile
from io import BytesIO
import hashlib
import threading
import time
import numpy as np

curDir=os.path.dirname(os.path.abspath(__file__))
print(curDir)
 
class StandardizationTests(unittest.TestCase):
    referenceFilePath=curDir+"/output/smiles_prepared.csv"

    ############################
    #### setup and teardown ####
    ############################
 
    # executed after each test
    def tearDown(self):
        pass
 
    def defineConfig(self):
        tuner.config.parameters.get_parameters(path=curDir+"/../tests/structure_preparation_test/example_parameters.json")
###############
#### tests ####
###############
 
    def test_structure_standardization_single(self):
        """Testing standardization of a single smiles from Chembl with reference smiles output obtained Feb 14th 2020"""
        self.defineConfig()
        response=chem_utils.structure_standardization("Cc1ccc(cc1)S(=O)(=O)Nc2ccc(cc2)c3nc4ccc(NS(=O)(=O)c5ccc(C)cc5)cc4[nH]3")
        self.assertEqual(response, "Cc1ccc(S(=O)(=O)Nc2ccc(-c3nc4cc(NS(=O)(=O)c5ccc(C)cc5)ccc4[nH]3)cc2)cc1")
    
    def test_run_standardize(self):
        """Testing standardization of smiles using threading"""
        ih=open(curDir+"/input/smiles.csv","r")
        smiles=[line.strip() for line in ih.readlines()]
        ih.close()
        response2=chem_utils.run_standardize(smiles,2)
        response4=chem_utils.run_standardize(smiles,4)
        #print(len(response2))
        #print(len(response4))
        self.assertCountEqual(response2,response4)
        

    def test_structure_standardization(self):
        """
        Testing standardization of a larger set of smiles from Chembl using serial execution
        Compared are resulting output files.
        """
        
        self.defineConfig()
        referenceFilePath=curDir+"/output/smiles_prepared.csv"
        tempFilePath=curDir+"/output/tmp/smiles_out.csv"
        ih=open(curDir+"/input/smiles.csv","r")
        #oh=open("unit_test/output/smiles_prepared.csv","w")    #-- used to write the reference data
        oh=open(tempFilePath,"w")
        for smi in ih.readlines():
            response=chem_utils.structure_standardization(smi.strip())
            oh.write(response)
            oh.write("\n")
        ih.close()
        oh.close()
        result=filecmp.cmp(referenceFilePath,tempFilePath, shallow=False)
        os.remove(tempFilePath)
        self.assertEqual(result,True)

    def test_run_fingerprint(self):
        """Test to build fingerprints from prepared smiles"""

        self.defineConfig()
        tempFilePath=curDir+"/output/tmp/fp_out.npy"
        fpReferenceFilePath=curDir+"/output/test_run_fingerprint_ref.npy"
        with open(self.referenceFilePath,"r") as h:
            smiles=[line.strip() for line in h.readlines()]
            fps2=chem_utils.run_fingerprint(smiles,2)
            fps4=chem_utils.run_fingerprint(smiles,4)
            #np.save("unit_test/output/test_run_fingerprint_ref.npy",fps2)   #write reference fingperprints
            np.save(tempFilePath,fps2)   #write reference fingperprints
            result=filecmp.cmp(fpReferenceFilePath,tempFilePath, shallow=False)
            np.save(tempFilePath,fps4)   #write reference fingperprints
            result2=filecmp.cmp(fpReferenceFilePath,tempFilePath, shallow=False)
            os.remove(tempFilePath)
            self.assertEqual(result and result2,True)

#if __name__ == "__main__":
    #unittest.main()
