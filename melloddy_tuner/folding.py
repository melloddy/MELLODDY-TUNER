import tqdm

from .chem_utils import make_desc_dict
from .config import parameters
from .helper import *


class ReadConfig(object):
    """
    Read parameters from config file.
    """

    def __init__(self, conf_dict=None):
        if conf_dict is None:
            conf_dict = {}
        self._conf_dict = conf_dict
        self._param = self._load_conf_dict()

    def get_conf_dict(self, conf_dict=None):
        if conf_dict is None:
            conf_dict = {}
        self._conf_dict = self._conf_dict if conf_dict is None else conf_dict
        return self._load_conf_dict()

    def _load_conf_dict(self):
        tmp_dict = self._conf_dict
        return tmp_dict


class LSHFolding(object):
    """Class to perform Local Sensitivity Hashing-based Folding"""

    def __init__(self):
        param = ReadConfig()
        lsh_param = param.get_conf_dict(parameters.get_parameters())['lsh']
        self.nfolds = lsh_param['nfolds']
        self.chembl_bits_list = lsh_param['bits']
        self.secret = param.get_conf_dict(parameters.get_parameters())['key']['key']

    def calc_highest_entropy_bits(self, ecfp):
        """
        Calculate highest entropy  (most informative) bits from given structure file.
        :param ecfp: fingerprints of given structure file
        :return: lists of highest entropy bits
        """
        ecfp_feat, ecfp_val = make_desc_dict(ecfp)

        # Create csr matrix based on fingerprint features and frequencies to obtain high entropy bits
        ecfp_csr, ecfp_feat_uniq = make_csr(ecfp_feat, ecfp_val)
        ecfp_csr.data = ecfp_csr.data.astype(np.int64)
        ecfp_csr_mean = np.array(ecfp_csr.mean(0)).flatten()

        ## filtering, and sorting based on distance from 0.5
        top10pct = np.where((ecfp_csr_mean < 0.9) & (ecfp_csr_mean > 0.1))[0]
        top10pct = top10pct[np.argsort(np.abs(0.5 - ecfp_csr_mean[top10pct]))]

        # Save high entropy bits
        df_high_entropy_bits = pd.DataFrame(
            {'ecfp': ecfp_feat_uniq[top10pct], 'ecfp_frequency': ecfp_csr_mean[top10pct]})

        return df_high_entropy_bits

    def run_lsh_calculation(self, ecfp):
        """
        Run LSH folding (clustering)
        :param ecfp: fingerprints of given structure file
        :return:    dataframe with fold id (cluster) for each entry
        """

        ecfp_feat, ecfp_val = make_desc_dict(ecfp)

        ## Load high entropy bits from ChEMBL
        lsh_chembl_16 = self.chembl_bits_list

        # check if bits are within the reference bit list
        lsh = [bits_to_str(np.isin(lsh_chembl_16, x)) for x in tqdm.tqdm(ecfp_feat)]
        lsh = np.array(lsh)

        # assign fold to lsh bit
        secret = self.secret
        folds = hashed_fold_lsh(lsh, secret.encode(), self.nfolds)

        df_folds = pd.DataFrame(folds)
        df_folds = df_folds.rename(columns={0: 'fold_id'})

        return df_folds
