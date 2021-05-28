import hashlib
import random
import hmac
import os

from rdkit import Chem
from rdkit.Chem.Scaffolds import rdScaffoldNetwork, MurckoScaffold
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import PandasTools
import argparse
import json
import pandas as pd

from rdkit import rdBase

rdBase.DisableLog("rdApp.*")


class ScaffoldFoldAssign(object):
    priority_cols = [
        "num_rings_delta",
        "has_macrocyle",
        "num_rbonds",
        "num_bridge",
        "num_spiro",
        "has_unusual_ring_size",
        "num_hrings",
        "num_arings",
        "node_smiles",
    ]
    priority_asc = [True, False, True, False, False, False, False, True, True]
    assert len(priority_cols) == len(
        priority_asc
    ), "priority_cols and priorty_asc must have same length"
    nrings_target = 3

    # rdScaffoldNetwork.ScaffoldNetworkParams are hardwired, since the heuristcs are not guaranteed to work with different setup here
    snparams = rdScaffoldNetwork.ScaffoldNetworkParams()
    snparams.flattenIsotopes = True
    snparams.includeGenericBondScaffolds = False
    snparams.includeGenericScaffolds = False
    snparams.includeScaffoldsWithAttachments = False  # this needs to be hardwired to False, as we start from Murcko, which has no attachment information
    snparams.includeScaffoldsWithoutAttachments = True  # this needs to hardwred to True,  as we start from Murcko, which has no attachment information
    snparams.pruneBeforeFragmenting = True

    # default constructor expecting all attributes passed as keyword arguments
    def __init__(self, secret, nfolds=5, verbosity=0):
        """Function to create and initialize a SaccolFoldAssign Calculator'

        Args:
            secret: secret key used (for fold hashing)
            nfolds: desired number of folds
            verbosity: controlls verbosity
        """

        self.nfolds = nfolds

        self.secret = secret.encode()
        self.verbosity = verbosity

    # methods required to pickle.
    # rdScaffoldNetwork.ScaffoldNetworkParams() canbnot be pickled and need to be initialized a new each time
    # def __getstate__(self):
    #    return self.secret, self.nfolds, self.verbosity

    # def __setstate__(self, secret, nfolds, verbosity):
    #    self.__init__(secret, nfolds, verbosity)

    @classmethod
    def from_param_dict(cls, secret, method_param_dict, verbosity=0):
        """Function to create and initialize a SaccolFoldAssign Calculator

        Args:
            secret: secret key used (for fold hashing)
            verbosity (int): controlls verbosity
            par_dict(dict): dictionary of method parameters
        """
        return cls(secret=secret, **method_param_dict, verbosity=verbosity)

    @staticmethod
    def murcko_scaff_smiles(mol_smiles):
        """Function to clauclate the Murcko scaffold, wrapper around rdkit MurckoScaffold.GetScaffoldForMol

        Args:
            mol_smiles(str): valid smiles of a molecule

        Returns:
            str: smiles string of the Murcko Scaffold

        """
        mol = Chem.MolFromSmiles(mol_smiles)
        if mol is not None:
            murcko_smiles = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
            if murcko_smiles == "":
                return None
            else:
                return murcko_smiles
        else:
            raise ValueError("could not parse smiles {}".format(mol_smiles))

    @staticmethod
    def has_unusual_ringsize(mol):
        """Function to check for ringsizes different than 5 or 6

        Args:
            mol(rdkit.Chem.rdchem.Mol): molecule

        Returns:
            bool: boolean indicating whether usnusally sized ring is present

        """

        return (
            len(
                [
                    len(x)
                    for x in mol.GetRingInfo().AtomRings()
                    if len(x) > 6 or len(x) < 5
                ]
            )
            > 0
        )

    @staticmethod
    def has_macrocycle(mol):
        """Function to check for macrocycles with rinsize > 9

        Args:
            mol(rdkit.Chem.rdchem.Mol): molecule

        Returns:
            bool: boolean indicating whether macrocycle is present

        """

        return len([len(x) for x in mol.GetRingInfo().AtomRings() if len(x) > 9]) > 0

    def sn_scaff_smiles(self, murcko_smiles):
        """Function to exctract the preferred scaffold based on Scaffold Tree rules from the scaffold network created from a Murcko scaffold

        Args:
            murcko_smiles(str): valdi smiles string of a Murcko scaffold

        Returns:
            str: smiles string of the preferred scaffold

        """

        if murcko_smiles is None:
            return None
        mol = Chem.MolFromSmiles(murcko_smiles)
        if mol is not None:
            # if the murcko scaffold has less or equal than the targeted number of rings, then the Murcko scaffold is already the sn_scaffold,
            # so no further decomposition is needed
            if Chem.rdMolDescriptors.CalcNumRings(mol) <= self.nrings_target:
                return murcko_smiles
            # otherwise start decomposition
            try:
                sn = rdScaffoldNetwork.CreateScaffoldNetwork([mol], self.snparams)
            except:
                raise ValueError(
                    "failed to calculate scaffold network for {}".format(murcko_smiles)
                )
            # create data fram with n ode smiles
            node_df = pd.DataFrame({"node_smiles": [str(n) for n in sn.nodes]})
            PandasTools.AddMoleculeColumnToFrame(
                node_df, "node_smiles", "mol", includeFingerprints=False
            )
            node_df["num_rings"] = node_df["mol"].apply(
                Chem.rdMolDescriptors.CalcNumRings
            )
            node_df["num_rings_delta"] = (
                node_df["num_rings"] - self.nrings_target
            ).abs()
            node_df["num_rbonds"] = node_df["mol"].apply(
                Chem.rdMolDescriptors.CalcNumRotatableBonds, strict=False
            )
            node_df["num_hrings"] = node_df["mol"].apply(
                Chem.rdMolDescriptors.CalcNumHeterocycles
            )
            node_df["num_arings"] = node_df["mol"].apply(
                Chem.rdMolDescriptors.CalcNumAromaticRings
            )
            node_df["num_bridge"] = node_df["mol"].apply(
                Chem.rdMolDescriptors.CalcNumBridgeheadAtoms
            )
            node_df["num_spiro"] = node_df["mol"].apply(
                Chem.rdMolDescriptors.CalcNumSpiroAtoms
            )
            node_df["has_macrocyle"] = node_df["mol"].apply(self.has_macrocycle)
            node_df["has_unusual_ring_size"] = node_df["mol"].apply(
                self.has_unusual_ringsize
            )
            node_df.sort_values(
                self.priority_cols, ascending=self.priority_asc, inplace=True
            )
            return node_df.iloc[0]["node_smiles"]
        else:
            raise ValueError(
                "murcko_smiles {} cannot be read by rdkit".format(murcko_smiles)
            )

    def hashed_fold_scaffold(self, sn_smiles):
        """applies hashing to assign scaffold sn_smiles to a fold

        Args:
            sn_smiles(str): smiles of the scaffold network scaffold

        Returns:
            int: fold id
        """
        scaff = str(sn_smiles).encode("ASCII")
        h = hmac.new(self.secret, msg=scaff, digestmod=hashlib.sha256)
        random.seed(h.digest(), version=2)
        return random.randint(0, self.nfolds - 1)

    # this function contaisn  the key functionality
    def calculate_single(self, smiles):
        """Function to calculate a sn_scaffold and fold_id from an individual smiles

        Args:
            smiles (str) : standardized smiles

        Returns:
            Tuple(str, str, int, bool, str) : a tuple of murcko_smiles, sn_scaffold_smiles, fold_id, Success_flag, error_message
        """
        try:
            murcko_smiles = self.murcko_scaff_smiles(smiles)
            sn_smiles = self.sn_scaff_smiles(murcko_smiles)
            fold_id = self.hashed_fold_scaffold(sn_smiles)
        except ValueError as err:
            return None, None, None, False, str(err)
        return murcko_smiles, sn_smiles, fold_id, True, None
