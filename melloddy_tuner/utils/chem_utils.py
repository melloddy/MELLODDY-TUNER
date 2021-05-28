import importlib
import json
import copy
import time
from typing import Tuple
from pandas.core.frame import DataFrame


from rdkit import rdBase
from rdkit.Chem.MolStandardize.tautomer import TautomerTransform
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit import Chem

rdBase.DisableLog("rdApp.*")
import logging
from multiprocessing import Pool
import tqdm
from rdkit.Chem import MolToSmiles, MolFromSmiles
from rdkit.Chem.AllChem import GetHashedMorganFingerprint, GetMorganFingerprint
import numpy as np

from .helper import *
from .config import ConfigDict, SecretDict


# set_start_method('fork')


def structure_standardization(smi: str) -> str:
    """
    Standardization function to clean up smiles with RDKit. First, the input smiles is converted into a mol object.
    Not-readable SMILES are written to the log file. The molecule size is checked by the number of atoms (non-hydrogen).
    If the molecule has more than 100 non-hydrogen atoms, the compound is discarded and written in the log file.
    Molecules with number of non-hydrogen atoms <= 100 are standardized with the MolVS toolkit
    (https://molvs.readthedocs.io/en/latest/index.html) relying on RDKit. Molecules which failed the standardization
    process are saved in the log file. The remaining standardized structures are converted back into their canonical
    SMILES format.
    :param smi: Input SMILES from the given structure data file T4
    :return: smi_clean: Cleaned and standardized canonical SMILES of the given input SMILES.


    Args:
        smi (str): Non-standardized smiles string

    Returns:
        str: standardized smiles string
    """

    # tautomer.TAUTOMER_TRANSFORMS = update_tautomer_rules()
    # importlib.reload(MolVS_standardizer)
    # param = ReadConfig()
    standardization_param = ConfigDict.get_parameters()["standardization"]

    max_num_atoms = standardization_param["max_num_atoms"]
    max_num_tautomers = standardization_param["max_num_tautomers"]
    include_stereoinfo = standardization_param["include_stereoinfo"]

    ## Load new tautomer enumarator/canonicalizer
    tautomerizer = rdMolStandardize.TautomerEnumerator()
    tautomerizer.SetMaxTautomers(max_num_tautomers)
    tautomerizer.SetRemoveSp3Stereo(
        False
    )  # Keep stereo information of keto/enol tautomerization

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

    mol = MolFromSmiles(smi)  # Read SMILES and convert it to RDKit mol object.
    if (
        mol is not None
    ):  # Check, if the input SMILES has been converted into a mol object.
        if (
            mol.GetNumAtoms() <= max_num_atoms
        ):  # check size of the molecule based on the non-hydrogen atom count.
            try:

                mol = rdMolStandardize.ChargeParent(
                    mol
                )  # standardize molecules using MolVS and RDKit
                mol = isotope_parent(mol)
                if include_stereoinfo is False:
                    Chem.RemoveStereochemistry(mol)
                    mol = tautomerizer.Canonicalize(mol)
                    mol_clean = my_standardizer(mol)
                    smi_clean = MolToSmiles(
                        mol_clean
                    )  # convert mol object back to SMILES
                else:
                    mol = tautomerizer.Canonicalize(mol)
                    mol_clean = my_standardizer(mol)
                    smi_clean = MolToSmiles(mol_clean)
            except (ValueError, AttributeError) as e:
                smi_clean = np.nan
                logging.error(
                    "Standardization error, " + smi + ", Error Type: " + str(e)
                )  # write failed molecules during standardization to log file

        else:
            smi_clean = np.nan
            logging.error("Molecule too large, " + smi)

    else:
        smi_clean = np.nan
        logging.error("Reading Error, " + smi)

    return smi_clean


def run_standardize(smi: list, num_cpu: int = 1) -> list:
    """
    Main function to run the standardization protocol in multiprocessing fashion. Standardization of molecules is
    distributed onto the given number of CPUs. The standardization script returns a csv file with the IDs and
    the standardized SMILES ('canonical_smiles') and a log file of the non-processed structures which failed for
    certain reasons.

    Args:
        smi (list): input smiles list
        num_cpu (int): number of CPU cores to use during multiprocessing (default=1)

    Returns:
        list: list of standardized smiles
    """
    start = time.time()
    with Pool(processes=num_cpu) as pool:
        ####################################
        smi_standardized = list(
            tqdm.tqdm(pool.imap(multi_standardization, smi), total=len(smi))
        )
        print(
            f"Standardizing {len(smi)} molecules took {time.time() - start:.08} seconds."
        )
        ####################################
        pool.close()
        pool.join()
    return smi_standardized


def multi_standardization(x: str) -> str:
    """
    Wrapper for mapping during multiprocessing RDKit standardization
    """
    return structure_standardization(x)


def output_processed_smiles(
    smi_standardized: list, df_structure: DataFrame
) -> DataFrame:
    """
    Add standardized smiles to structure dataframe.

    Args:
        smi_standardized (list): list of standardized smiles
        df_structure (DataFrame): structure dataframe

    Returns:
        DataFrame: structure dataframe including column of standardized smiles
    """
    df_clean_structures = df_structure
    df_clean_structures["canonical_smiles"] = smi_standardized

    df_clean_structures.dropna(subset=["canonical_smiles"], inplace=True)

    return df_clean_structures


def output_failed_smiles(smi_standardized: list, df_structure: DataFrame) -> DataFrame:
    """
        Output of structure failed during the standardization process.


    Args:
        smi_standardized (list): list of standardized smiles
        df_structure (DataFrame): structure dataframe

    Returns:
        DataFrame: structure dataframe with entries failed during standardization.
    """
    df_clean_structures = df_structure
    df_clean_structures["canonical_smiles"] = smi_standardized
    df_failed_structures = df_clean_structures[
        pd.isna(df_structure["canonical_smiles"])
    ]
    return df_failed_structures


def fp_calc(smi: str) -> Tuple:
    """
    Calculation of Morgan fingerprints (ECFP equivalent) with a given radius

    Args:
        smi (str): SMILES string

    Returns:
        Tuple(np.array(list), np.array(list)): NumPy arrays of fingerprint feature list, fingerprint value list
    """
    fingerprint_param = ConfigDict.get_parameters()["fingerprint"]
    fp_radius = fingerprint_param["radius"]
    fp_hashed = fingerprint_param["hashed"]
    fp_nbits = fingerprint_param["fold_size"]

    mol_fp = {}
    mol = MolFromSmiles(smi)

    if mol is not None:
        if fp_hashed is True:
            try:
                mol_fp = GetHashedMorganFingerprint(
                    mol, fp_radius, fp_nbits
                ).GetNonzeroElements()
            except:
                mol_fp = np.nan
                logging.error("Fingerprint calculation error, " + smi)
        else:
            try:
                mol_fp = GetMorganFingerprint(mol, fp_radius).GetNonzeroElements()
            except:
                mol_fp = np.nan
                logging.error("Fingerprint calculation error, " + smi)
    return np.array(list(mol_fp.keys())), np.array(list(mol_fp.values()))


def multi_fp_calc(x: str) -> Tuple:
    """
    Wrapper for multiprocessing fingerprint calculations
    """
    return fp_calc(x)


def run_fingerprint(smi: list, num_cpu: int = 1) -> list:
    """
        Calculation of Morgan fingerprints with a multiprocessing setup. The standardized SMILES strings are distributed onto
    the given number of CPUs.

    Args:
        smi (list): list of smiles strings.
        num_cpu (int, optional): Number of CPU cores to use during multiprocessing. Defaults to 1.

    Returns:
        list: List of fingerprint tuples (features, values)
    """
    with Pool(processes=num_cpu) as pool:
        ####################################
        start = time.time()
        ####################################
        mol_fp = list(tqdm.tqdm(pool.imap(multi_fp_calc, smi), total=len(smi)))
        print(
            f"Calculating fingerprints of {len(smi)} molecules took {time.time() - start:.08} seconds."
        )
        pool.close()
        pool.join()
    return mol_fp

    """
    Converts fingerprint features and values into a scrambled lists of fingerprint features.
    :param ecfp_feat:   Fingerprint features
    :param ecfp_val:    Fingerprint values
    :return:
    """


def make_fp_lists(ecfp_feat: dict, ecfp_val: dict) -> Tuple:
    """
    Convert fingerprint feature and value dictionaries into lists of scrambled features and non-scrambled values

    Args:
        ecfp_feat (dict): dictionary of fingerprint features
        ecfp_val (dict): dictionary of fingeprint values

    Returns:
        Tuple (list, list): scrambled feature list, non-scrambled value list
    """

    ecfp_feat_list = list(ecfp_feat)

    ecfp_val_list = list(ecfp_val)
    # param = ReadConfig()
    fingerprint_param = ConfigDict.get_parameters()["fingerprint"]
    fp_binarized = fingerprint_param["binarized"]
    fp_bitsize = fingerprint_param["fold_size"]
    key_settings = SecretDict.get_secrets()
    secret = key_settings["key"]
    ecfp_feat_list_scrambled = make_scrambled_lists(ecfp_feat_list, secret, fp_bitsize)

    if fp_binarized:
        for e in ecfp_val_list:
            e.fill(1)
    return ecfp_feat_list_scrambled, ecfp_val_list

    """
    Save the fingerprint information in the given structure data frame.
    :param ecfp: Dictionary of fingerprint features (keys) and values (values)
    :param df_structure: given structure dataframe
    :return: DataFrame with descriptor id and fingerprint information
    """


def output_processed_descriptors(ecfp, df_structure):
    """
    Save the fingerprint information in the given structure data frame.
    :param ecfp: Dictionary of fingerprint features (keys) and values (values)
    :param df_structure: given structure dataframe
    :return: Dataframe with descriptor id and fingerprint information
    """
    # Convert list of tuples back to dictionary of fingerprint features and frequencies
    ecfp_feat, ecfp_val = make_desc_dict(ecfp)

    ecfp_feat_list_scrambled = make_fp_lists(ecfp_feat, ecfp_val)[0]

    ecfp_val_list = make_fp_lists(ecfp_feat, ecfp_val)[1]
    # adding fingerprint information to dataframe columns
    df_structure["ecfp_feat"] = ecfp_feat_list_scrambled
    df_structure["ecfp_val"] = ecfp_val_list
    # create fingerprint column in json format to identify duplicates
    df_structure["fp_feat"] = [arr.tolist() for arr in ecfp_feat_list_scrambled]
    df_structure["fp_val"] = [arr.tolist() for arr in ecfp_val_list]
    df_structure["fp_feat"] = df_structure["fp_feat"].apply(lambda x: json.dumps(x))
    df_structure["fp_val"] = df_structure["fp_val"].apply(lambda x: json.dumps(x))

    # identify duplicated fingerprint, create unique descriptor vector ID for them,
    # and sort them according to the new descriptor ID
    return df_structure


def make_desc_dict(ecfp):
    """
    Format tuple of fingerprint information into dictionary
    :param ecfp: Tuple of fingerprint features and values
    :return: dictionary of fingerprint features and values
    """
    ecfp_feat, ecfp_val = zip(*ecfp)
    return ecfp_feat, ecfp_val


def output_descriptor_duplicates(df_structure):
    """
    Output duplicate entries based on descriptor id.
    :param df_structure: Given dataframe with fingerprint information and descriptor id
    :return: duplicate descriptor ids
    """
    df_structure_duplicates = pd.DataFrame(
        df_structure,
        columns=[
            "input_compound_id",
            "canonical_smiles",
            "fp_feat",
            "descriptor_vector_id",
        ],
    )
    df_structure_duplicates = df_structure_duplicates[
        df_structure_duplicates.duplicated(["descriptor_vector_id"], keep=False)
    ]
    df_structure_duplicates = df_structure_duplicates.sort_values(
        "descriptor_vector_id"
    )
    return df_structure_duplicates


def output_mapping_table(
    df_structure: DataFrame, col_to_keep: list = None
) -> DataFrame:
    """
    Generate mapping table to map original and new descriptor ids.

    Args:
        df_structure (DataFrame): structure dataframe
        col_to_keep (list, optional): list of columns appearing in the output dataframe. Defaults to None.

    Returns:
        DataFrame: mapping table
    """
    if col_to_keep is None:
        col_to_keep = df_structure.columns
    mapping_table = pd.DataFrame(df_structure, columns=col_to_keep)
    mapping_table = mapping_table.sort_values(mapping_table.columns[0])

    return mapping_table
