from math import nan
from typing import Tuple
from rdkit import rdBase
from rdkit.Chem.MolStandardize.tautomer import TautomerTransform
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Chem

rdBase.DisableLog("rdApp.*")
from rdkit.Chem import MolToSmiles, MolFromSmiles
from rdkit.Chem.AllChem import GetHashedMorganFingerprint, GetMorganFingerprint

from .helper import *

# from .config import ConfigDict, SecretDict
import copy


class Standardizer(object):
    def __init__(
        self, max_num_atoms, max_num_tautomers, include_stereoinfo, verbosity=0
    ):

        self.max_num_atoms = max_num_atoms
        self.max_num_tautomers = max_num_tautomers
        self.include_stereoinfo = include_stereoinfo
        self.verbosity = verbosity

        ## Load new tautomer enumarator/canonicalizer
        self.tautomerizer = rdMolStandardize.TautomerEnumerator()
        self.tautomerizer.SetMaxTautomers(self.max_num_tautomers)
        self.tautomerizer.SetRemoveSp3Stereo(
            False
        )  # Keep stereo information of keto/enol tautomerization

    # functions enabling pickle
    def __getstate__(self):
        return (
            self.max_num_atoms,
            self.max_num_tautomers,
            self.include_stereoinfo,
            self.verbosity,
        )

    def __setstate__(self, state):
        self.__init__(*state)

    @classmethod
    def from_param_dict(cls, method_param_dict, verbosity=0):
        """Function to create and initialize a Standardizer Calculator

        Args:
            verbosity (int): controlls verbosity
            par_dict(dict): dictionary of method parameters
        """
        return cls(**method_param_dict, verbosity=verbosity)

    @staticmethod
    def my_standardizer(mol: Chem.Mol) -> Chem.Mol:
        """
        MolVS implementation of standardization

        Args:
            mol (Chem.Mol): non-standardized rdkit mol object

        Returns:
            Chem.Mol: stndardized rdkit mol object
        """
        mol = copy.deepcopy(mol)
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        disconnector = rdMolStandardize.MetalDisconnector()
        mol = disconnector.Disconnect(mol)
        normalizer = rdMolStandardize.Normalizer()
        mol = normalizer.normalize(mol)
        reionizer = rdMolStandardize.Reionizer()
        mol = reionizer.reionize(mol)
        Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
        # TODO: Check this removes symmetric stereocenters
        return mol

    @staticmethod
    def isotope_parent(mol: Chem.Mol) -> Chem.Mol:
        """
        Isotope parent from MOLVS
        Return the isotope parent of a given molecule.
        The isotope parent has all atoms replaced with the most abundant isotope for that element.
        Args:
            mol (Chem.Mol): input rdkit mol object

        Returns:
            Chem.Mol: isotope parent rdkit mol object
        """
        mol = copy.deepcopy(mol)
        # Replace isotopes with common weight
        for atom in mol.GetAtoms():
            atom.SetIsotope(0)
        return mol

    def calculate_single(self, smiles: str) -> Tuple:
        try:
            smi_clean = self.calculate_single_raising(smiles)
            return smi_clean, True, None
        except (TypeError, ValueError, AttributeError) as e:
            return None, False, str(e)

    def calculate_single_raising(self, smiles: str) -> str:
        """
        This function stanrdardizes a single smiles 
        """
        if smiles is nan:
            error = "No smiles entry."
            raise ValueError(error)
        # Read SMILES and convert it to RDKit mol object.
        mol = MolFromSmiles(smiles) 
        if mol is None:
            raise ValueError("failed to parse smiles {}".format(smiles))
        # check size of the molecule based on the non-hydrogen atom count.
        if mol.GetNumAtoms() >= self.max_num_atoms:
            error = "number of non-H atoms {0} exceeds limit of {1} for smiles {2}".format(mol.GetNumAtoms(), self.max_num_atoms, smiles)
            raise ValueError(error)
        # standardize molecules using MolVS and RDKit
        mol = rdMolStandardize.ChargeParent(mol)  
        mol = self.isotope_parent(mol)
        if self.include_stereoinfo is False:
            Chem.RemoveStereochemistry(mol)
        mol = self.tautomerizer.Canonicalize(mol)
        mol_clean_tmp = self.my_standardizer(mol)
        smi_clean_tmp = MolToSmiles(
            mol_clean_tmp
        )  # convert mol object back to SMILES
        ## Double check if standardized SMILES is a valid mol object
        mol_clean = MolFromSmiles(smi_clean_tmp)
        smi_clean = MolToSmiles(mol_clean)
        return smi_clean