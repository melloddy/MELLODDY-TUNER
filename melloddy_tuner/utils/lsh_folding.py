import json
from math import nan

import pandas as pd
from melloddy_tuner.utils.folding import LSHFolding
from melloddy_tuner.utils.helper import bits_to_str, hashed_fold_lsh, int_to_sha256
from typing import Dict, List, Tuple
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetHashedMorganFingerprint, GetMorganFingerprint

import numpy as np
import copy
import logging


class LSHFoldingCalculator(object):
    """
    Wrapper class to calculate molecular descriptors (fingerprints).


    """

    def __init__(
        self,
        radius: int,
        hashed: bool,
        fold_size: int,
        binarized: bool,
        nfolds: int,
        bits: list,
        secret: str,
        verbosity: int = 0,
    ):

        self.radius = radius
        self.hashed = hashed
        self.size = fold_size
        self.binarize = binarized
        self.nfolds = nfolds
        self.chembl_bits_list = bits
        self.secret = secret
        self.verbosity = verbosity
        self.permuted_ix = LSHFoldingCalculator.set_permutation(
            size=self.size, key=self.secret
        )

    # functions enabling pickle
    def __getstate__(self):
        return (
            self.radius,
            self.hashed,
            self.size,
            self.binarize,
            self.nfolds,
            self.chembl_bits_list,
            self.secret,
            self.verbosity,
        )

    def __setstate__(self, state):
        self.__init__(*state)

    @classmethod
    def from_param_dict(cls, secret, method_param_dict, verbosity=0):
        """Function to create and initialize a LSHFolding Calculator

        Args:
            secret: secret key used (for fold hashing)
            verbosity (int): controlls verbosity
            par_dict(dict): dictionary of method parameters
        """
        return cls(secret=secret, **method_param_dict, verbosity=verbosity)

    def set_permutation(size: int, key: str):
        original_ix = np.arange(size)
        hashed_ix = np.array(
            [
                int.from_bytes(int_to_sha256(j, key), "big") % 2 ** 63
                for j in original_ix
            ]
        )
        permuted_ix = hashed_ix.argsort().argsort()
        return permuted_ix

    def make_scrambled_lists(self, fp_feat_arr: np.array) -> list:
        """Pseudo-random scrambling with secret.

        Args:
            fp_list (list): fingerprint list
            secret (str): secret key
            bitsize (int): bitsize (shape)

        Returns:
            list: scrambled list
        """
        original_ix = np.arange(self.size)

        scrambled = []
        if (np.sort(self.permuted_ix) == original_ix).all():
            for x in fp_feat_arr:
                scrambled.append(int(self.permuted_ix[x]))
        else:
            print("Check index permutation failed.")
        return np.array(scrambled)

    def get_fp(self, smiles):
        mol_fp = {}

        mol = MolFromSmiles(smiles)  # Read SMILES and convert it to RDKit mol object.

        if self.hashed is True:
            try:
                mol_fp = GetHashedMorganFingerprint(
                    mol, self.radius, self.size
                ).GetNonzeroElements()
            except (ValueError, AttributeError) as e:
                return None, False, str(e)
            return mol_fp
        else:
            try:
                mol_fp = GetMorganFingerprint(
                    mol, self.radius, self.size
                ).GetNonzeroElements()
            except (ValueError, AttributeError) as e:
                return None, False, str(e)
            return mol_fp

    def scramble_fp(self, mol_fp):
        fp_feat = np.array(list(mol_fp.keys()))
        fp_val = np.array(list(mol_fp.values()))
        fp_feat_scrambled = LSHFoldingCalculator.make_scrambled_lists(self, fp_feat)
        if self.binarize:
            fp_val.fill(int(1))

        return fp_feat_scrambled.tolist(), fp_val.tolist()

    @staticmethod
    def fp_to_json(fp_feat, fp_val):
        fp_feat_json = json.dumps(fp_feat)
        fp_val_json = json.dumps(fp_val)
        return fp_feat_json, fp_val_json

    def run_lsh_calculation(self, ecfp_feat: List) -> pd.DataFrame:
        """
        Run LSH folding (clustering)

        Args:
            ecfp (Tuple):  lists fingerprint  features and values

        Returns:
            DataFrame: dataframe containing fold assignments
        """

        # ecfp_feat = make_desc_dict(ecfp)[0]
        ## Load high entropy bits from ChEMBL
        lsh_chembl_16 = self.chembl_bits_list

        # check if bits are within the reference bit list
        lsh = bits_to_str(np.isin(lsh_chembl_16, ecfp_feat))
        lsh = np.array(lsh)
        # assign fold to lsh bit
        secret = self.secret
        folds = hashed_fold_lsh(lsh, secret.encode(), self.nfolds)
        # df_folds = pd.DataFrame(folds)
        # df_folds = df_folds.rename(columns={0: 'fold_id'})
        return np.int(folds)

    def calculate_single(self, smiles: str) -> Tuple:
        """
        Calculation of Morgan fingerprints (ECFP equivalent) with a given radius

        Args:
            smi (str): SMILES string

        Returns:
            Tuple(np.array(list), np.array(list)): NumPy arrays of fingerprint feature list, fingerprint value list
        """

        try:
            mol_fp = self.get_fp(smiles)
        except ValueError as err:
            return None, None, None, False, str(err)
        try:
            fp_feat_raw = np.array(list(mol_fp.keys()))
            fold_id = self.run_lsh_calculation(fp_feat_raw.tolist())
            fp_feat_scrambled, fp_val_binarized = self.scramble_fp(mol_fp)
            fp_feat, fp_val = self.fp_to_json(fp_feat_scrambled, fp_val_binarized)

        except ValueError as err:
            return None, None, None, False, str(err)

        return fp_feat, fp_val, fold_id, True, None
