from melloddy_tuner.utils.standardizer import Standardizer
from melloddy_tuner.utils.descriptor_calculator import DescriptorCalculator
import json
import os
from scipy.sparse import csr_matrix
import torch


#json string with structure praration paramaters used in the MELLODDY federated models
melloddy_structure_prep_params = """
{
  "standardization": {
    "max_num_tautomers": 256,
    "max_num_atoms": 100,
    "include_stereoinfo": false
  },
  "fingerprint": {
    "radius": 3,
    "hashed": true,
    "fold_size": 32000,
    "binarized": true
  }
}
"""


def process_param_input(param_input):
    """
    This function acceptsy variobale types of paramter input and returns the results as dict
    1. if the indupt is a dictionary, it is handed through directly
    2. If the input is a str, it is first ested whether the input is a valid file name. 
       If this is the case it will be first attempted to read  the file in as a json file
    3. If the input is a string, and it is not a valid filename, it will pe attempted to parse this as a json string
    
    """
    if type(param_input) == dict:
        return(param_input)
    elif type(param_input) == str:
        if os.path.isfile(param_input):
            with open(param_input) as cnf_f:
                tmp_dict = json.load(cnf_f)
                return tmp_dict
        else:
            try:
                tmp_dict = json.loads(param_input)
                return tmp_dict
            except json.JSONDecodeError as e:
                raise ValueError("The string input for paramaters \"{}\" is neither a valid filename nor a valid json string".format(param_input))
    else:
        raise ValueError("The paramater input is neither a dictionary nor a json string or a valid filename")
                



class SingleRowPreparator:
    """
    This wrapper class contains all functionality to generate prediction ready features for sparsechem prediction for individual compounds as torch sparse coo-tensor
    """
    
    
    def __init__(self, params, secret, trust_standardization = False, verbosity = 0):
        """
        Initialize the single row preparator
        
        Args:
            params: parameter information for stnadardization and fingerprint calculation 
                    (dictionary, json encoded dictionary, or the path of a file containing a json encoded dictionary 
            secret: key information constaining the fingeprint permutationkey 
                    (dictionary, json encoded dictionary, or the path of a file containing a json encoded dictionary)
            trust_standardization (bool): Flag whether to assume a smiles input is already standardized
        """
        my_params = process_param_input(params)
        my_secret = process_param_input(secret)
        self.trust_standardization = trust_standardization
        if (not "standardization" in my_params) or (not "fingerprint" in my_params):    
            raise ValueError("The provided parameters does not contain the required keys \"standardization\" and \"fingerprint\" ")
        if not "key" in my_secret:
            raise ValueError("The provided secret dictionary does not contain the required keys \"key\" ")

        
        if not self.trust_standardization:
            self.standardizer = Standardizer.from_param_dict(my_params["standardization"], verbosity = verbosity)
        self.descriptor_calc = DescriptorCalculator.from_param_dict(my_secret["key"],my_params["fingerprint"], verbosity = verbosity)
        
        
    def process_smiles(self, smiles : str) -> torch.tensor:
        """
        This function stnadrdaizes a single smiles and computes the fingeprrint features as torch sparse coo-tensor
        
        Args:
             smiles: smiles of the molecule to claculet features for
	 
        Returns:
            torch.tesnor: A torch sparse coo tebnsor of fingeprrint features
        """
        if not self.trust_standardization:
            smiles = self.standardizer.calculate_single_raising(smiles)
        return self.descriptor_calc.calculate_single_torch_coo(smiles)
