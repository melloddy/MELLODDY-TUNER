import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import hashlib
import random
import hmac


class ReadConfig(object):
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



def read_csv(file, delimiter=',', chunksize=None, nrows=None):
    """
    Read a comma-separated file with structural data.
    :param file: path of the input file
    :param delimiter: comma as delimiter
    :return: pandas dataframe of the input data
    """
    df_input = pd.read_csv(file,
                           delimiter=delimiter, chunksize=chunksize, nrows=nrows)
    return df_input


def bits_to_str(bits):
    return "".join(str(int(x)) for x in bits)



def make_csr(ecfpx, ecfpx_counts):
    """
    create a csr (compressed sparse row) matrix
    :param ecfpx: fingerprint features of the given structure data
    :param ecfpx_counts: fingerprint feature frequencies
    :return: the csr matrix and a unique array of fingerprint features
    """
    ecfpx_lengths = [len(x) for x in ecfpx]
    ecfpx_cmpd = np.repeat(np.arange(len(ecfpx)), ecfpx_lengths)
    ecfpx_feat = np.concatenate(ecfpx)
    ecfpx_val = np.concatenate(ecfpx_counts)

    ecfpx_feat_uniq = np.unique(ecfpx_feat)
    fp2idx = dict(zip(ecfpx_feat_uniq, range(ecfpx_feat_uniq.shape[0])))
    ecfpx_idx = np.vectorize(lambda i: fp2idx[i])(ecfpx_feat)

    X0 = csr_matrix((ecfpx_val, (ecfpx_cmpd, ecfpx_idx)))
    return X0, ecfpx_feat_uniq



def make_scrambled_lists(fp_list, secret, bitsize):
    """
    Args:
      ecfp_list   list of ECFP np.arrays
      secret      secret for hashing 
    Returns list of scrambled ECFP lists (arrays)
    """
    original_ix = np.arange(bitsize)
    hashed_ix = np.array([int.from_bytes(int_to_sha256(j, secret), "big") % 2**63 for j in original_ix])
    permuted_ix = hashed_ix.argsort().argsort()
    if (np.sort(permuted_ix) == original_ix).all():
        scrambled = []
        for x in fp_list:
            scrambled.append(permuted_ix[list(x)])
    else:
        print('Check index permutation failed.')
    return scrambled
    
    
def make_lsh(X6, bits):
    """
    :param X6: csr matrix of the fingerprint
    :param bits:  bits given as fixed parameter. length default = 16
    :return: local sensitivity hashes
    """
    bit2int = np.power(2, np.arange(len(bits)))
    lsh = X6[:, bits] @ bit2int
    return lsh

def int_to_sha256(i, secret):
    """HMAC for converting integer i to hash, using SHA256 and secret."""
    return hmac.new(secret.encode('utf-8'), str(i).encode('utf-8'), hashlib.sha256).digest()


def sha256(inputs, secret):
    """
    Encryption function using python's pre-installed packages HMAC and hashlib.
    We are using SHA256 (it is equal in security to SHA512).
    :param inputs: input strings
    :param secret: given pharma partner key
    :return:
    """
    m = hmac.new(secret, b'', hashlib.sha256)
    for i in inputs:
        m.update(i)
    return m.digest()


def lsh_to_fold(lsh, secret, nfolds):
    """
    use encryption to secure lsh to folds
    :param lsh: LSH cluster
    :param secret: given pharma-only key
    :param nfolds: number of folds as defined in the parameter file. default = 5
    :return: fold_id of LSH cluster
    """
    lsh_bin = str(lsh).encode("ASCII")
    h = sha256([lsh_bin], secret)
    random.seed(h, version=2)
    return random.randint(0, nfolds - 1)


def hashed_fold_lsh(lsh, secret, nfolds=5):
    """

    :param lsh:
    :param secret: given pharma-only key
    :param nfolds: number of folds as defined in the parameter file. default = 5
    :return:
    """
    lsh_uniq = np.unique(lsh)
    lsh_fold = np.vectorize(lambda x: lsh_to_fold(x, secret, nfolds=nfolds))(lsh_uniq)
    lsh2fold = dict(zip(lsh_uniq, lsh_fold))
    return np.vectorize(lambda i: lsh2fold[i])(lsh)



