from typing import Tuple
from pandas.core.frame import DataFrame
import tqdm

from .chem_utils import make_desc_dict
from .helper import *
from .config import ConfigDict, SecretDict


class LSHFolding(object):

    """Class to perform Local Sensitivity Hashing-based Folding"""

    def __init__(self):
        lsh_param = ConfigDict.get_parameters()["lsh"]
        self.nfolds = lsh_param["nfolds"]
        self.chembl_bits_list = lsh_param["bits"]
        self.secret = SecretDict.get_secrets()["key"]

    @staticmethod
    def calc_highest_entropy_bits(ecfp: Tuple) -> DataFrame:
        """
        Calculate highest entropy  (most informative) bits from given structure file.

        Args:
            ecfp (Tuple): fingerprint lists of features and values

        Returns:
            DataFrame: dataframe containing high entropy bits of given fingerprints.
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
            {
                "ecfp": ecfp_feat_uniq[top10pct],
                "ecfp_frequency": ecfp_csr_mean[top10pct],
            }
        )

        return df_high_entropy_bits

    def run_lsh_calculation(self, ecfp: Tuple) -> DataFrame:
        """
        Run LSH folding (clustering)

        Args:
            ecfp (Tuple):  lists fingerprint  features and values

        Returns:
            DataFrame: dataframe containing fold assignments
        """

        ecfp_feat = make_desc_dict(ecfp)[0]

        ## Load high entropy bits from ChEMBL
        lsh_chembl_16 = self.chembl_bits_list

        # check if bits are within the reference bit list
        lsh = [bits_to_str(np.isin(lsh_chembl_16, x)) for x in tqdm.tqdm(ecfp_feat)]
        lsh = np.array(lsh)

        # assign fold to lsh bit
        secret = self.secret
        folds = hashed_fold_lsh(lsh, secret.encode(), self.nfolds)

        df_folds = pd.DataFrame(folds)
        df_folds = df_folds.rename(columns={0: "fold_id"})

        return df_folds
