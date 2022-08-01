
from datetime import datetime
import json
import jsonschema
from jsonschema import validate
import os
from typing import Dict, Tuple, List
import glob
from numpy.core.defchararray import array
import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema, Index, Category

import numpy as np
from pandas.core.frame import DataFrame
from scipy.sparse import csr_matrix, save_npz
import hashlib
import random
import hmac
from pathlib import Path
from .config import secrets, parameters
import logging


# class ReadConfig(object):
#     """
#     Config Class


#     """
#     def __init__(self, conf_dict: dict = None):
#         """
#         Initialize config class.

#         Args:
#             conf_dict (dict, optional): Config dictionary containig parameters. Defaults to None.
#         """
#         if conf_dict is None:
#             conf_dict = {}
#         self._conf_dict = conf_dict
#         self._param = self._load_conf_dict()

#     def get_conf_dict(self, conf_dict: dict =None):
#         """
#         Get config dictionary.

#         Args:
#             conf_dict (dict, optional): config dictionary. Defaults to None.


#         """
#         if conf_dict is None:
#             conf_dict = {}
#         self._conf_dict = self._conf_dict if conf_dict is None else conf_dict
#         return self._load_conf_dict()

#     def _load_conf_dict(self):
#         """
#         Load config dictionary.

#         """
#         tmp_dict = self._conf_dict
#         return tmp_dict




def load_config(args: dict):
    """
    Load config from file.

    Args:
        args (dict): argparser arguments .

    """
    config_file = Path(args["config_file"])
    if config_file.is_file() is False:
        print("Config file does not exist.")
        return quit()

    if parameters.get_config_file() != config_file:
        parameters.__init__(config_path=config_file)
        return print("Read new config file.")


def load_config_from_dict(config: dict):
    """
    Load config from a dictionary.

    Args:
        config(dict): config dictionary.
    """
    parameters.__init__()
    parameters._CONFIG_FILE = ""
    parameters._CONFIG = config


def load_key(args: dict):
    """
    Load key from file.

    Args:
        args (dict): argparser arguments .

    """
    key_file = Path(args["key_file"])
    if key_file.is_file() is False:
        print("Key file does not exist.")
        return quit()

    if secrets.get_key_file() != key_file:
        secrets.__init__(key_path=key_file)
        return print("Read new key file.")


def load_key_from_dict(key: dict):
    """
    Load key from a dictionary.

    Args:
        key(dict): key dictionary.
    """
    secrets.__init__()
    secrets._KEY_FILE = ""
    secrets._KEY = key


def read_input_file(path: str) -> DataFrame:
    """
    Read csv file and return dataframe.

    Args:
        path (str): [input path to csv file

    Returns:
        DataFrame: pandas dataframe from csv file
    """
    data_file = Path(path)
    if data_file.is_file() is False:
        print(f"{data_file} does not exist.")
        return quit()
    df = read_csv(data_file)
    input_file_len = len(df)
    if input_file_len == 0:
        print(f"Input file {path} is empty. Please provide a suitable structure file.")
        return quit()
    return df


def read_input_files(paths: List[str]) -> List[DataFrame]:
    """
    Read csv files and return a list of dataframes.

    Args:
        path (List[str]): input paths to csv files

    Returns:
        List[DataFrame]: pandas dataframes from csv file
    """
    return [read_input_file(file) for file in paths]


def save_df_as_csv(
        output_dir: Path, df: DataFrame, name: str = None, col_to_keep: list = None
):
    """
    Save dataframe to csv file.

    Args:
        df (DataFrame): provided dataframe
        output_dir (Path): path to output folder
        name (str, optional): filename. Defaults to None.
        col_to_keep (list, optional): columns to write into the output file. Defaults to None.
    """
    if col_to_keep is None:
        col_to_keep = df.columns
    path = output_dir / f"{name}.csv"
    df.to_csv(path, sep=",", columns=col_to_keep, index=False)

def sanity_check_input_assay_id(T0_list: List[DataFrame]):
    dict_input_assay = {
        "T0_duplicates": {}
    }
    flags = []
    if len(T0_list)>1:
        common_iai = set.intersection(*map(set, [T0['input_assay_id'] for T0 in T0_list]))
        if len(common_iai) != 0:
            print(f'Error: T0 files must contain distinct input assays ids, found {len(common_iai)} compounds in common :')
            flag = "ERROR"
            dict_input_assay["T0_duplicates"]["FLAG"] = flag
            dict_input_assay["T0_duplicates"]["num_duplicates"] = len(common_iai)
            
        else:
            flag = "OK"
            dict_input_assay["T0_duplicates"]["FLAG"] = flag
            dict_input_assay["T0_duplicates"]["num_duplicates"] = len(common_iai)
        flags.append(flag)
        passed =  "ERROR" not in flags
        return passed, dict_input_assay
    else:
        return sanity_check_uniqueness(T0_list[0], "input_assay_id", "T0")
        

def sanity_check_assay_type(T0: DataFrame, tag):

    dict_assay_type = {
        "non_standard_types": {},
        "auxiliary_assay_types": {},
        "missing_types": {},
        "unique_assay_count":{},
        "empty_types":{}
    }
    flags = []

    std_types = ["ADME", "NON-CATALOG-PANEL", "CATALOG-PANEL", "OTHER", "AUX_HTS", "AUX_PL"]

    assay_types_counts = T0["assay_type"].value_counts()
    found_types = T0["assay_type"].unique().tolist()
    
    # look for non standard assay type names
    for at in found_types:
        if at not in std_types:
            print(f"WARNING: found non standard assay type name: {at}")
            flag = "WARNING"
        else:
            flag = "OK"
        dict_assay_type["non_standard_types"]["FLAG"] = flag
        dict_assay_type["non_standard_types"]["assay_type"] = at
        flags.append(flag)

    # look for auxiliary types depending on tag
    if tag == "no":
        if "AUX_HTS"  in found_types or "AUX_PL" in found_types:
            print(f"ERROR: Found auxiliary assay types in T0.")
            flag = "ERROR"
        else:
            flag = "OK"
        dict_assay_type["auxiliary_assay_types"]["using_auxiliary"] = tag
        dict_assay_type["auxiliary_assay_types"]["FLAG"] = flag
        dict_assay_type["auxiliary_assay_types"]["info"] = "Auxiliary assay types are found in non-auxiliary setup."
        flags.append(flag)
    if tag == "yes":
        if "AUX_HTS" not in found_types and "AUX_PL" not in found_types:
            print(f"ERROR: Missing auxiliary assay types in T0.")
            flag = "ERROR"
        else:
            flag = "OK"
        dict_assay_type["auxiliary_assay_types"]["using_auxiliary"] = tag
        dict_assay_type["auxiliary_assay_types"]["FLAG"] = flag
        dict_assay_type["auxiliary_assay_types"]["info"] = "Check missing types below."
        flags.append(flag)
    # look for missing assay types
    if len(assay_types_counts.index) < 6:
        found_types = T0["assay_type"].unique().tolist()
        diff_types = list(set(std_types) - set(found_types))
        print(f"INFO: missing assay type: {diff_types}")
        flag = "INFO"
        dict_assay_type["missing_types"]["FLAG"] = flag
        dict_assay_type["missing_types"]["missing_types"] = diff_types
        flags.append(flag)
    if len(assay_types_counts.index) > 6:
        print(f"ERROR : too many assay type, found : {len(assay_types_counts.index)}")
        flag = "ERROR"
    else:
        flag = "OK"

    flags.append(flag) 
    dict_assay_type["unique_assay_count"]["FLAG"] = flag
    dict_assay_type["unique_assay_count"]["assay_count"] = len(assay_types_counts.index)

    # look for empty assay type values
    if T0.loc[T0["assay_type"].isna()].shape[0]:
        print("ERROR: assay types has missing values")
        flag = "ERROR"
    else:
        flag = "OK"
    flags.append(flag)
    dict_assay_type["empty_types"]["FLAG"] = flag
    dict_assay_type["empty_types"]["number_empty_types"] = T0.loc[T0["assay_type"].isna()].shape[0]
    passed =  "ERROR" not in flags
    return passed ,dict_assay_type

def sanity_check_assay_sizes(T0, T1):
    dict_assay_sizes = {
        "not_in_T1": {},
        "not_in_T0": {}
    }
    flags = []
    t0_assays = T0.input_assay_id.unique()
    t1_assays = T1.input_assay_id.unique()
    num_t0_assay = t0_assays.shape[0]
    num_t1_assay = t1_assays.shape[0]

    if num_t1_assay < num_t0_assay:
        print("WARNING : T1 does not have all input_assay_id present in T0")
        assays_not_in_t1 = T0.loc[~T0["input_assay_id"].isin(t1_assays)]
        flag =  "WARNING"
        dict_assay_sizes["not_in_T1"]["FLAG"] = flag
        dict_assay_sizes["not_in_T1"]["num_assay"] = assays_not_in_t1.input_assay_id.shape[0]
        flags.append(flag)
    if num_t0_assay < num_t1_assay:
        print("ERROR: some input_assay_id present in T1 are not present in T0")
        assays_not_in_t0 = T1.loc[~T1["input_assay_id"].isin(t0_assays)]
        flag =  "ERROR"
        dict_assay_sizes["not_in_T0"]["FLAG"] = flag
        dict_assay_sizes["not_in_T0"]["num_assay"] = assays_not_in_t0.input_assay_id.shape[0]
        flags.append(flag)
    passed =  "ERROR" not in flags
    return passed, dict_assay_sizes
    


def allign_compound_sizes(T1, T2):
    if not sanity_check_compound_sizes(T1, T2)[0]:
        cleaned_t2 = T2[T2.input_compound_id.isin(T1.input_compound_id)]
        t2_failed = T2[~T2.input_compound_id.isin(T1.input_compound_id)]
        return cleaned_t2, t2_failed
    return T2, pd.DataFrame()


def sanity_check_compound_sizes(T1, T2):

    dict_compound_sizes = {
        "cmpds_not_T1": {},
        "cmpds_not_T2": {}
    }
    flags = []

    t1_compounds = T1.input_compound_id.unique()
    t2_compounds = T2.input_compound_id.unique()
    num_t2_compounds = t2_compounds.shape[0]
    num_t1_compounds = t1_compounds.shape[0]

    if num_t1_compounds < num_t2_compounds:
        print("WARNING : T1 does not contain data for all input_compound_id in T2")
        compounds_not_in_t1 = T2.loc[~T2["input_compound_id"].isin(t1_compounds)]
        print("Compounds not in T1:")
        print(compounds_not_in_t1.input_compound_id.nunique())
        flag = "WARNING"
        dict_compound_sizes["cmpds_not_T1"]["FLAG"] = flag
        dict_compound_sizes["cmpds_not_T1"]["number"] = compounds_not_in_t1.input_compound_id.nunique()
        flags.append(flag)
        
    if num_t2_compounds < num_t1_compounds:
        print("**** ERROR: some input_compound_id present in T1 are not present in T2")
        compounds_not_in_t2 = T1.loc[~T1["input_compound_id"].isin(t2_compounds)]
        print("Number of compounds not in T2:")
        print(compounds_not_in_t2.input_compound_id.nunique())
        flag = "ERROR"
        dict_compound_sizes["cmpds_not_T2"]["FLAG"] = flag
        dict_compound_sizes["unique_cmpds_not_T2"]["number"] = compounds_not_in_t2.input_compound_id.nunique()
        flags.append(flag)
    passed =  "ERROR" not in flags
    return passed, dict_compound_sizes


def sanity_check_uniqueness(df, colname, filename):
    # verif input_compound_id duplicates
    dict_uniqueness= {
        f"{filename}": {}
    }
    flags = []
    dict_uniqueness[f"{filename}"]["colname"] = colname
    duplic = df[colname].duplicated(keep=False)
    if duplic.sum() > 0:
        print(
            f"Found {duplic.sum()} records with *{colname}* present multiple times in {filename}."
        )
        print(df[duplic])

    df_nrows = df.shape[0]
    df_id = df[colname].nunique()
    delta = abs(df_nrows-df_id)
    dict_uniqueness[f"{filename}"]["num_non_unique"] = delta
    if df_nrows != df_id:
        flag = "ERROR"


    else:
        flag = "OK"
    flags.append(flag)
    passed =  "ERROR" not in flags
    dict_uniqueness[f"{filename}"]["FLAG"] = flags
    return passed, dict_uniqueness

def assays_per_type (df: DataFrame, mode: str):
    if mode == "clf":
        assays_count = df[df.cont_classification_task_id.notna()].groupby("assay_type").agg(
            n_assays=pd.NamedAgg(column="input_assay_id", aggfunc = "nunique")
            )
    elif mode ==  "reg":
        assays_count = df[df.cont_regression_task_id.notna()].groupby("assay_type").agg(
            n_assays=pd.NamedAgg(column="input_assay_id", aggfunc = "nunique")
            )
    return assays_count

def tasks_per_type (df: DataFrame, type: str):
    if type == "clf":
        tasks_count = df.groupby("assay_type").agg(
            n_tasks=pd.NamedAgg(column="cont_classification_task_id", aggfunc = "count")
            ).reset_index()
    elif type ==  "reg":
        tasks_count = df.groupby("assay_type").agg(
            n_tasks=pd.NamedAgg(column="cont_regression_task_id", aggfunc = "count")
            ).reset_index()
    return tasks_count

def counts_per_type(df: DataFrame, mode: str, type: str):
    if mode == "tasks":
        df_counts = tasks_per_type(df, type)
    elif mode == "assays":
        df_counts = assays_per_type(df, type)
    elif mode == "both":
        df_tasks = tasks_per_type(df, type)
        df_assays = assays_per_type(df, type)
        df_counts = pd.merge(df_tasks, df_assays, on="assay_type")
    return df_counts

def sanity_check_binary(df: DataFrame):
    dict_binary = {}
    flags = []
    if df.loc[df.assay_type == "AUX_PL", "is_binary"].all():
        flag =  "OK"
    else: 
        flag =  "ERROR"
    flags.append(flag)
    if df.loc[df.is_binary == True, "use_in_regression"].any():
        flag = "WARNING"
    else:
        flag = "OK"
    flags.append(flag)
    passed =  "ERROR" not in flags
    dict_binary["assays_binary_regression_mismatch"] = df.loc[(df.is_binary == True) & (df["use_in_regression"]), "input_assay_id"].unique()
    return passed, dict_binary

    

def assays_per_type (df: DataFrame, mode: str):
    if mode == "clf":
        assays_count = df[df.cont_classification_task_id.notna()].groupby("assay_type").agg(
            n_assays=pd.NamedAgg(column="input_assay_id", aggfunc = "nunique")
            )
    elif mode ==  "reg":
        assays_count = df[df.cont_regression_task_id.notna()].groupby("assay_type").agg(
            n_assays=pd.NamedAgg(column="input_assay_id", aggfunc = "nunique")
            )
    return assays_count

def tasks_per_type (df: DataFrame, type: str):
    if type == "clf":
        tasks_count = df.groupby("assay_type").agg(
            n_tasks=pd.NamedAgg(column="cont_classification_task_id", aggfunc = "count")
            ).reset_index()
    elif type ==  "reg":
        tasks_count = df.groupby("assay_type").agg(
            n_tasks=pd.NamedAgg(column="cont_regression_task_id", aggfunc = "count")
            ).reset_index()
    return tasks_count

def counts_per_type(df: DataFrame, mode: str, type: str):
    if mode == "tasks":
        df_counts = tasks_per_type(df, type)
    elif mode == "assays":
        df_counts = assays_per_type(df, type)
    elif mode == "both":
        df_tasks = tasks_per_type(df, type)
        df_assays = assays_per_type(df, type)
        df_counts = pd.merge(df_tasks, df_assays, on="assay_type")
    return df_counts


def save_mtx_as_npy(matrix: csr_matrix, output_dir: Path, name: str = None):
    """
    Save csr matrix as npy matrix.

    Args:
        matrix (csr_matrix): input csr matrix
        output_dir (Path): output path
        name (str, optional): filename. Defaults to None.
    """

    if "T11_fold_vector" in name:
        path = output_dir / f"{name}.npy"
        np.save(path, matrix)
    else:
        # path_npy = output_dir / f'{name}.npy'
        # np.save(path_npy, matrix)
        path = output_dir / f"{name}.npz"
        save_npz(path, matrix)


def concat_desc_folds(df_desc: DataFrame, df_folds: DataFrame) -> DataFrame:
    """
    Concatenate descriptor and fold dataframes.

    Args:
        df_desc (DataFrame): descriptor dataframe
        df_folds (DataFrame): fold dataframe

    Returns:
        DataFrame: concatenated dataframe
    """
    df_out = pd.concat([df_desc, df_folds], axis=1)
    df_out = df_out.drop_duplicates(
        ["descriptor_vector_id", "fp_feat", "fp_val", "fold_id"]
    ).sort_values("descriptor_vector_id")
    return df_out


def make_results_dir(args: dict, overwriting: bool) -> Path:
    """
    Create results folder

    Args:
        args (dict): argparser arguments
        overwriting (bool): overwriting option

    Returns:
        Path: Results folder path
    """
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(exist_ok=True)
    run_name = args["run_name"]
    dir_run_name = output_dir / run_name
    dir_run_name.mkdir(exist_ok=True)
    path_results_extern = dir_run_name / "results"
    path_results_extern.mkdir(exist_ok=True)
    if overwriting is False:
        if os.listdir(path_results_extern):
            override = input(
                f"Do you want to override files in {dir_run_name}? (type y or Y) \n The script will be aborted if you type anything else. "
            )
            if override == "y" or override == "Y":
                print(f"Files for run name {run_name} will be overwritten.")
            else:
                print(
                    "Processing aborted. Please change the run name and re-run the script."
                )
                quit()
    return path_results_extern


def make_ref_dir(args: dict, overwriting: bool) -> Path:
    """
    Create reference output subfolder.

    Args:
        args (dict): argparser arguments
        overwriting (bool): overwriting option

    Returns:
        Path: reference output path
    """
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(exist_ok=True)
    run_name = args["run_name"]
    dir_run_name = output_dir / run_name
    dir_run_name.mkdir(exist_ok=True)
    output_dir_name = dir_run_name / "reference_set"
    output_dir_name.mkdir(exist_ok=True)
    if overwriting is False:
        if os.listdir(output_dir_name):
            override = input(
                f"Do you want to override files in {dir_run_name}? (type y or Y) \n The script will be aborted if you type anything else. "
            )
            if override == "y" or override == "Y":
                print(f"Files for run name {run_name} will be overwritten.")
            else:
                print(
                    "Processing aborted. Please change the run name and re-run the script."
                )
                quit()
    return output_dir_name


def make_dir(args: dict, subdir: str, name: str, overwriting: bool) -> Path:
    """
    Creates a folder in the defined subfolder with the given name.

    Args:
        args (dict): argparser arguments
        subdir (str): subfolder name
        name (str): folder name
        overwriting (bool): overwriting option

    Returns:
        Path: path the created directory
    """
    output_dir = Path(args["output_dir"])
    output_dir.mkdir(exist_ok=True)
    run_name = args["run_name"]
    dir_run_name = output_dir / run_name
    dir_run_name.mkdir(exist_ok=True)
    output_dir_name = dir_run_name
    if subdir is not None:
        output_dir_name = dir_run_name / subdir
        output_dir_name.mkdir(exist_ok=True)
    if name is not None:
        output_dir_name = output_dir_name / f"{name}"
        output_dir_name.mkdir(exist_ok=True)
    if overwriting is False:
        if os.listdir(output_dir_name):
            override = input(
                f"Do you want to override files in {dir_run_name}? (type y or Y) \n The script will be aborted if you type anything else. "
            )
            if override == "y" or override == "Y":
                print(f"Files for run name {run_name} will be overwritten.")
            else:
                print(
                    "Processing aborted. Please change the run name and re-run the script."
                )
                quit()
    return output_dir_name


def create_log_files(output_dir: Path):
    """
    Create log files.

    Args:
        arg (dict): argparser arguments
        output_dir (Path): path of output folder
    """
    name = output_dir.name
    log_file_path = output_dir / f"{name}.log"
    logging.basicConfig(
        filename=log_file_path, filemode="w", format="", level=logging.ERROR
    )


def read_csv(
        file: Path, delimiter: str = ",", chunksize: int = None, nrows: int = None
) -> DataFrame:
    """
    Read a comma-separated file with structural data.

    Args:
        file (Path): file path
        delimiter (str, optional): delimiter. Defaults to ','.
        chunksize (int, optional): cunksize. Defaults to None.
        nrows (int, optional): number of rows to read. Defaults to None.

    Returns:
        DataFrame: pandas dataframe
    """
    df_input = pd.read_csv(file, delimiter=delimiter, chunksize=chunksize, nrows=nrows)
    return df_input


def format_dataframe(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Create unique descriptor vector id and generate mapping table T5 and save duplicates.

    Args:
        df (DataFrame): dataframe with descriptor features and values, as well as input compound id.

    Returns:
        Tuple[Dataframe, Dataframe, DataFrame]: T5 mapping table, T6_table, descriptor-based duplicates
    """
    # identify duplicated fingerprint, create unique descriptor vector ID for them,
    # and sort them according to the new descriptor ID
    df["descriptor_vector_id"] = df.groupby(["fp_feat", "fp_val", "fold_id"]).ngroup()
    # extract the mapping table before duplicate checking
    df_T5 = df[["input_compound_id", "fold_id", "descriptor_vector_id"]]
    # we sort now, as we need to have sorted datframes for T6 and the duplicates
    df = df.sort_values("descriptor_vector_id")
    # we identify duplciates
    # duplicate removal based desriptor vector ID is sufficient, becuase it implies unique 'fp_feat', 'fp_val', 'fold_id' combinations
    df_T6 = df.drop_duplicates("descriptor_vector_id")[
        ["descriptor_vector_id", "fp_feat", "fp_val", "fold_id"]
    ]
    is_duplicated = df.duplicated(["descriptor_vector_id"], keep=False)
    df_duplicates = df.loc[
        is_duplicated,
        [
            "input_compound_id",
            "canonical_smiles",
            "fp_feat",
            "fp_val",
            "fold_id",
            "descriptor_vector_id",
        ],
    ]

    return df_T5, df_T6, df_duplicates


def bits_to_str(bits: list) -> str:
    """
    convert bits to string

    Args:
        bits (list): bits as list

    Returns:
        str: bit list string
    """
    return "".join(str(int(x)) for x in bits)


def make_csr(ecfpx: dict, ecfpx_counts: dict) -> Tuple:
    """
    create a csr (compressed sparse row) matrix from fingerprint dictionaries


    Args:
        ecfpx (dict): fingerprint features
        ecfpx_counts (dict): fingerprint values

    Returns:
        Tuple: csr matrix and NumPy array of unique features.
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


def make_scrambled_lists(fp_list: list, secret: str, bitsize: int) -> list:
    """
    Pseudo-random scrambling with secret.

    Args:
        fp_list (list): fingerprint list
        secret (str): secret key
        bitsize (int): bitsize (shape)

    Returns:
        list: scrambled list
    """
    original_ix = np.arange(bitsize)
    hashed_ix = np.array(
        [int.from_bytes(int_to_sha256(j, secret), "big") % 2 ** 63 for j in original_ix]
    )
    permuted_ix = hashed_ix.argsort().argsort()
    scrambled = []
    if (np.sort(permuted_ix) == original_ix).all():

        for x in fp_list:
            scrambled.append(permuted_ix[list(x)])
    else:
        print("Check index permutation failed.")
    return scrambled


# Function to generate LSH of certain bit size. Currently not used.
# def make_lsh(X6, bits):
#     """
#     :param X6: csr matrix of the fingerprint
#     :param bits:  bits given as fixed parameter. length default = 16
#     :return: local sensitivity hashes
#     """
#     bit2int = np.power(2, np.arange(len(bits)))
#     lsh = X6[:, bits] @ bit2int
#     return lsh


def map_2_cont_id(data: DataFrame, colname: str) -> DataFrame:
    """
    Mapping function to get continuous identifiers

    Args:
        data (DataFrame): dataframe with ID column
        colname (str): name of ID column

    Returns:
        DataFrame: mapped dataframe
    """
    map_id = {val: ind for ind, val in enumerate(np.unique(data[colname]))}
    map_id_df = pd.DataFrame.from_dict(map_id, orient="index").reset_index()
    map_id_df = map_id_df.rename(columns={"index": colname, 0: "cont_" + colname})
    data_remapped = pd.merge(data, map_id_df, how="inner", on=colname)
    return data_remapped


def int_to_sha256(i: int, secret: str) -> str:
    """
    HMAC for converting integer i to hash, using SHA256 and secret.

    Args:
        i (int): integer
        secret (str): secret key

    Returns:
        str: HMAC encoded string
    """

    return hmac.new(
        secret.encode("utf-8"), str(i).encode("utf-8"), hashlib.sha256
    ).digest()


def sha256(inputs: list, secret: str) -> str:
    """
    SHA256 hashing of list with secret key

    Args:
        inputs (list): list of input hashes
        secret (str): secret key

    Returns:
        str: HMAC encoded string
    """
    m = hmac.new(secret, b"", hashlib.sha256)
    for i in inputs:
        m.update(i)
    return m.digest()


def lsh_to_fold(lsh: int, secret: str, nfolds: int = 5) -> int:
    """
    use encryption to secure lsh to folds

    Args:
        lsh (int): lsh integer
        secret (str): secret key
        nfolds (int): number of folds. default = 5

    Returns:
        int: pseudo-random number
    """
    lsh_bin = str(lsh).encode("ASCII")
    h = sha256([lsh_bin], secret)
    random.seed(h, version=2)
    return random.randint(0, nfolds - 1)


def hashed_fold_lsh(lsh: np.array, secret: str, nfolds: int = 5) -> np.array:
    """
    Get folds by hashing LSH strings.

    Args:
        lsh (np.array): Array of LSH strings
        secret (str): secret key
        nfolds (int, optional): number of folds. Defaults to 5.

    Returns:
        np.array: hashed folds
    """
    lsh_uniq = np.unique(lsh)
    lsh_fold = np.vectorize(lambda x: lsh_to_fold(x, secret, nfolds=nfolds))(lsh_uniq)
    lsh2fold = dict(zip(lsh_uniq, lsh_fold))
    return np.vectorize(lambda i: lsh2fold[i])(lsh)


def save_run_args(_args: dict, mode: str):
    output_dir = make_dir(_args, None, None, overwriting= True)
    remove_keys = ['func', "run_parameters"]
    run_params = dict((key, val) for key, val in _args.items() if key not in remove_keys)
    with open(f'{output_dir}/{mode}_run_params.json', 'wt') as f:
	    json.dump(run_params, f, indent=4)
	    f.close() 
    print(f"Wrote input params to '{output_dir}/{mode}_run_params.json'\n")
    return run_params

def save_run_report(_args: dict, dict_report: dict, mode: str):
    output_dir = make_dir(_args, None, None, overwriting= True)
    today = datetime.today().strftime('%Y-%m-%d')
    with open(f'{output_dir}/{today}-{mode}_run_report.json', 'wt') as f:
	    json.dump(dict_report, f, indent=4)
	    f.close() 
    print(f"Wrote run report to '{today}-{mode}_run_report.json'\n")


def read_run_params(path: str) -> dict:
    """
    Reads the json file, checks the given parameter types and parses them to Python types.

    :param path: Path to config file
    :return:params from json file
    """
    path_keys = ["file", "dir"]
    with open(path, 'r') as json_file:
        data = json.load(json_file)
        valid = validate_json(data)
        if valid:
            print('Reading run parameters from given json file')
            
            return data
        else:
            print(f"Non valid parameters in {path}")
            quit()

def validate_json(json_data: dict) -> bool:
    """Validate run parameters dictionary.

    Args:
        data (dict): run parameter dictionary

    Returns:
        bool: True =  Valid run parameters, False = not valid run parameters
    """
    main_location = os.path.dirname(os.path.realpath(__file__))
    default_config_path = os.path.join(
            main_location, "../data/"
        )
    with open(f"{default_config_path}run_params_schema.json", 'r') as json_file:
        run_param_schema = json.load(json_file)
    try:
        validate(instance=json_data, schema=run_param_schema)
    except jsonschema.exceptions.ValidationError as err:
        print(err)
        return False
    return True

def concatenate_T_files(T_df_list: List[DataFrame]):
    return pd.concat(T_df_list, axis = 0)

def validate_T0(T0: DataFrame):
    std_types = ["ADME", "NON-CATALOG-PANEL", "CATALOG-PANEL", "OTHER", "AUX_HTS", "AUX_PL"]

    schema = DataFrameSchema(
    {
        "input_assay_id": Column(int, unique=True),
        "assay_type": Column(
            Category,
            checks=Check.isin(std_types),
            coerce=True,
            regex=True,
            ),
        "use_in_regression": Column(bool),
        "is_binary": Column(bool),
        "expert_threshold_1": Column(float, nullable=True),
        "expert_threshold_2": Column(float, nullable=True),
        "expert_threshold_3": Column(float, nullable=True),
        "expert_threshold_4": Column(float, nullable=True),
        "expert_threshold_5": Column(float, nullable=True),
        "direction": Column(str, nullable=True),
        "catalog_assay_id": Column(float, nullable=True),
        "parent_assay_id": Column(float, nullable=True)
    },
    index=Index(int),
    strict=False,
    coerce=False,
    )
    
    schema_catalog_type = DataFrameSchema(
    {
        "assay_type": Column(
            str,
            Check(lambda s: s == "CATALOG-PANEL", element_wise=True)
        )

    },
    index=Index(int),
    strict=False,
    coerce=False,
    )
    schema_catalog_unqiue = DataFrameSchema(
    {
        "catalog_assay_id": Column(
            float,
            unique=True,
            nullable=True
        )

    },
    index=Index(int),
    strict=False,
    coerce=False,
    )
    try:
        schema.validate(T0, lazy=True)
        schema_catalog_type.validate(T0[T0.catalog_assay_id.notnull()], lazy=True)
        schema_catalog_unqiue.validate(T0[T0.assay_type == "CATALOG-PANEL"], lazy=True)
    except pa.errors.SchemaErrors as err:
        print("Schema errors and failure cases:")
        print(err.failure_cases)

        quit("T0 validation failed.")

def validate_T1(T1: DataFrame):
    schema = DataFrameSchema(
    {
        "input_compound_id": Column(int), 
        "input_assay_id": Column(int),
        "standard_qualifier": Column(str, nullable=True),
        "standard_value" : Column(float)
    },
    index=Index(int),
    strict=True,
    coerce=False,
    )
    try:
        schema.validate(T1, lazy=True)
    except pa.errors.SchemaErrors as err:
        print("Schema errors and failure cases:")
        print(err.failure_cases)

        quit("T1 validation failed.")

def validate_T2(T2: DataFrame):
    schema = DataFrameSchema(
    {
        "input_compound_id": Column(int, unique=True), 
        "smiles": Column(str)
    },
    index=Index(int),
    strict=False,
    coerce=False,
    )
    try:
        schema.validate(T2, lazy=True)
    except pa.errors.SchemaErrors as err:
        print("Schema errors and failure cases:")
        print(err.failure_cases)

        quit("T2 validation failed.")
