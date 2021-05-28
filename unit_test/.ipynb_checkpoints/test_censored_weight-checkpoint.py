import os
import unittest
import sys
from melloddy_tuner.utils.config import ConfigDict
from melloddy_tuner.scripts.filter_regression import censored_weight_transformation

import filecmp
from pathlib import Path
import pandas as pd
import numpy as np

curDir=os.path.dirname(os.path.abspath(__file__))
print(curDir)
 
class CensoredWeightTests(unittest.TestCase):
    

    ############################
    #### setup and teardown ####
    ############################
 
    # executed after each test
    def tearDown(self):
        pass
    
    
    def setUp(self):
        self.config = ConfigDict(config_path=Path(os.path.join(curDir,'reference_files','example_parameters.json'))).get_parameters()


###############
#### tests ####
###############

    #standalone test for censored weight calculation
    def test_censored_weight(self):
        fraction_censored = pd.Series([1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1,0.05,0.025,0.0])
        result_actual = censored_weight_transformation(fraction_censored,**self.config['censored_downweighting'])
        result_expected = pd.Series([0.0, 0.005847953216374268, 0.013157894736842103, 0.022556390977443615, 0.035087719298245626, 0.052631578947368425, 0.07894736842105263, \
                                     0.12280701754385964, 0.21052631578947373, 0.4736842105263158, 1.0, 1.0, 1.0])
        self.assertEqual(np.allclose(result_expected,result_actual,rtol=1e-05, atol=1e-08),True)