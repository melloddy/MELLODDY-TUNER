import json
from math import nan
from melloddy_tuner.utils.helper import int_to_sha256
from typing import Dict, Tuple
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.AllChem import GetHashedMorganFingerprint, GetMorganFingerprint
from scipy.sparse import csr_matrix
import torch

import numpy as np
import copy
import logging


class DescriptorCalculator(object):
    """
    Wrapper class to calculate molecular descriptors (fingerprints).


    """

    def __init__(
        self,
        radius: int,
        hashed: bool,
        fold_size: int,
        binarized: bool,
        secret: str,
        verbosity: int = 0,
    ):

        self.radius = radius
        self.hashed = hashed
        self.size = fold_size
        self.secret = secret
        self.binarize = binarized
        self.verbosity = verbosity
        self.permuted_ix = DescriptorCalculator.set_permutation(
            size=self.size, key=self.secret
        )

    # functions enabling pickle
    def __getstate__(self):
        return (
            self.radius,
            self.hashed,
            self.size,
            self.binarize,
            self.secret,
            self.verbosity,
        )

    def __setstate__(self, state):
        self.__init__(*state)

    @classmethod
    def from_param_dict(cls, secret, method_param_dict, verbosity=0):
        """Function to create and initialize a SaccolFoldAssign Calculator

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
        fp_feat_scrambled = DescriptorCalculator.make_scrambled_lists(self, fp_feat)
        if self.binarize:
            fp_val.fill(int(1))

        return fp_feat_scrambled, fp_val

    @staticmethod
    def fp_to_json(fp_feat, fp_val):
        fp_feat_json = json.dumps(fp_feat.tolist())
        fp_val_json = json.dumps(fp_val.tolist())
        return fp_feat_json, fp_val_json

    
    def get_scrambled_fp(self, smiles: str) -> Tuple:
        mol_fp = self.get_fp(smiles)
        fp_feat_scrambled, fp_val = self.scramble_fp(mol_fp)
        return fp_feat_scrambled, fp_val
    
    
    def calculate_single(self, smiles: str) -> Tuple:
        """
        Calculation of Morgan fingerprints (ECFP equivalent) with a given radius

        Args:
            smi (str): SMILES string

        Returns:
            Tuple(np.array(list), np.array(list)): NumPy arrays of fingerprint feature list, fingerprint value list
        """

        try:
            fp_feat_scrambled, fp_val = self.get_scrambled_fp(smiles)
            fp_feat, fp_val = self.fp_to_json(fp_feat_scrambled, fp_val)
        except ValueError as err:
            return None, None, False, str(err)
        return fp_feat, fp_val, True, None

    def calculate_single_csr(self, smiles: str) -> csr_matrix:
        """
        Calculation of Morgan fingerprints (ECFP equivalent) with a given radius as single row scipy csr matrix
        intended for directly passing to sparsechem

        Args:
            smi (str): SMILES string

        Returns:
            scipy csr matrix of fingerprint in the given bit width
        """
        fp_feat_scrambled, fp_val = self.get_scrambled_fp(smiles)
        row_ind = np.repeat(0, fp_feat_scrambled.shape[0])
        return csr_matrix((fp_val, (row_ind, fp_feat_scrambled)), shape = (1, self.size))
    
    
    def calculate_single_torch_coo(self, smiles: str) -> torch.sparse_coo_tensor :
        """
        Calculation of Morgan fingerprints (ECFP equivalent) with a given radius as single row torch sparse coo_tensor
        intended for directly passing to sparsechem

        Args:
            smiles (str): SMILES string

        Returns:
            torch sparse_coo_tensor of fingerprint in the given bit width
        """
        fp_feat_scrambled, fp_val = self.get_scrambled_fp(smiles)
        row_ind = np.repeat(0, fp_feat_scrambled.shape[0])
        return torch.sparse_coo_tensor(indices = np.array([row_ind,fp_feat_scrambled]), values = fp_val, size = (1, self.size),dtype=torch.float)