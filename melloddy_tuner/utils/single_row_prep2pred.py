from melloddy_tuner.utils import DescriptorCalculator, Standardizer
import json
import os
from scipy.sparse import csr_matrix


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
    This wrapper class contains all functionality to gnerate prediction ready features for sparsechem prediction for individual compounds
    
    """
    
    def __init__(self, params, secret, trust_standardization = False, verbosity = 0):
        
        my_params = process_param_input(params)
        if (not "standardization" in my_params) or (not "fingerprint" in my_params):    
            raise ValueError("The provided parameters does not contain the required keys \"standardization"\ and \"fingerprint\" ")
        my_secret = process_param_input(secret)
        if not key in my_secret:
            raise ValueError("The provided secret dictionary does not contain the required keys \"key\" ")
        self.trust_standardization = trust_standardization
        
        if not self.trust_standardization:
            self.standardizer = Standardizer.from_param_dict(my_params["standardization"], verbosity = verbosity)
        self.descriptor_calc = DescriptorCalculator.from_param_dict(my_secret[key],my_params["fingerprint"], verbosity = verbosity)
        
        
    def process_smiles(self, smiles : str) > csr_matrix:
        if not self.trust_standardization:
            smiles = self.standardizer.calculate_single(smiles)
        return self.descriptor_calc.calculate_single_csr(smiles)
