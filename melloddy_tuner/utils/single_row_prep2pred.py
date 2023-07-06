from melloddy_tuner.utils.standardizer import Standardizer
from melloddy_tuner.utils.descriptor_calculator import DescriptorCalculator
import json
import os
from scipy.sparse import csr_matrix
import torch
from pathlib import Path

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


def process_param_input(param_input = melloddy_structure_prep_params):
    """
    This function acceptsy variobale types of paramter input and returns the results as dict
    1. if the indupt is a dictionary, it is handed through directly
    2. If the input is a str, it is first ested whether the input is a valid file name. 
       If this is the case it will be first attempted to read  the file in as a json file
    3. If the input is a string, and it is not a valid filename, it will pe attempted to parse this as a json string
    4. If the param_input is None, it will return the default parameters (melloddy_structure_prep_params as above)
    """
    if isinstance(param_input, dict):
        return(param_input)
    elif isinstance(param_input, Path):
        with open(param_input) as cnf_f:
            tmp_dict = json.load(cnf_f)
            return tmp_dict
    elif isinstance(param_input, str):
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
                

# The key provider classes below can be used to provide the key to the predictor
# different versions below exist, depending on how the key shoudl be provided
#all key provider must implement the get_key function
class KeyProvider:
    """Abstract base class for provision of secret keys"""
    def get_key(self):
        """This function needs to be implemented by the derived class, and will be called by the single row preparator to apply the key"""
        raise NotImplementedError()
        
        
class KeyProviderFromJsonFile(KeyProvider):
    """This key provider stores a key file name and opens that file upon calling of get_key to read out the key """
    def __init__(self, key_file):
        if not os.path.isfile(key_file):
            raise FileNotFoundError("Key file {0} does not exists or is not rreadbyle".format(key_file))
        self.key_file = key_file
        
    def get_key(self):
        with open(self.key_file) as cnf_f:
            tmp_dict = json.load(cnf_f)
            return tmp_dict["key"]
            
class KeyProviderFromEnv(KeyProvider):
    """ This key provoder provides the key from an environment variable name. The key proivider store the name of the variable, and evlautes upon call of get_key"""
    def __init__(self, key_env_var):
        if not key_env_var in os.environ:
            raise ValueError("Environment variable {0} to hold the key is not defined",format(key_env_var))
        self.key_env_var = key_env_var
        
    def get_key(self):
        return os.environ[self.key_env_var]
    
    
class KeyProviderFromKeyValue(KeyProvider):
    """This key provider is initialized with the key value directly, which is tored within the object"""
    def __init__(self, key_value):
        self.key = key
        
    def get_key(self):
        return self.key
    
class KeyProviderTrivial(KeyProvider):
    """This class provides the trivial key for testing"""
    def get_key(self):
        return "melloddy"

        

class SingleRowPreparator:
    """
    This wrapper class contains all functionality to generate prediction ready features for sparsechem prediction for individual compounds as torch sparse coo-tensor
    """
    
    
    def __init__(self, key_provider : KeyProvider, params=melloddy_structure_prep_params, trust_standardization : bool = False, verbosity : int = 0):
        """
        Initialize the single row preparator
        
        Args:
            params: parameter information for stnadardization and fingerprint calculation 
                    (dictionary, json encoded dictionary, or the path of a file containing a json encoded dictionary 
            key_provider: Key Provider object providing the scarmbling key
            trust_standardization (bool): Flag whether to assume a smiles input is already standardized
        """
        my_params = process_param_input(params)
        self.key_provider = key_provider 
        self.trust_standardization = trust_standardization
        if (not "standardization" in my_params) or (not "fingerprint" in my_params):    
            raise ValueError("The provided parameters does not contain the required keys \"standardization\" and \"fingerprint\" ")

        
        if not self.trust_standardization:
            self.standardizer = Standardizer.from_param_dict(my_params["standardization"], verbosity = verbosity)
        self.descriptor_calc = DescriptorCalculator.from_param_dict(self.key_provider.get_key(),my_params["fingerprint"], verbosity = verbosity)
        
        
    def process_smiles(self, smiles : str) -> torch.Tensor:
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

    def apply_key(self):
        """
        This recovers te key from the key provider and applies it to the descriptor calculator
        Intented to be used when reloading a serialized SingleRowPreparator
        """
        self.descriptor_calc.apply_key(self.key_provider.get_key())
        
    def purge_key(self):
        """
        Delete the key information  and permutaion map in the descriptor calculator so that it caln be safely serialized without any key information stored within. 
        In this case at desiralization stage the key needs to re-applied with the apply key function
        """
        self.descriptor_calc.purge_key()
    