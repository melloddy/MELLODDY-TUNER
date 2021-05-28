import os
from typing import Tuple
from numpy.core.defchararray import array
import pandas as pd
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
        print("Structure input is empty. Please provide a suitable structure file.")
        return quit()
    return df


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


def sanity_check_assay_type(T0: DataFrame):

    std_types = ["ADME", "PANEL", "OTHER", "AUX_HTS"]
    assay_types_counts = T0["assay_type"].value_counts()
    found_types = T0["assay_type"].unique().tolist()
    # look for non standard assay type names
    for at in found_types:
        if at not in std_types:
            print(f"WARNING: found non standard assay type name: {at}")

    # look for missing assay types
    if len(assay_types_counts.index) < 4:
        found_types = T0["assay_type"].unique().tolist()
        diff_types = list(set(std_types) - set(found_types))
        print(f"INFO: missing assay type: {diff_types}")

    if len(assay_types_counts.index) > 4:
        exit(f"ERROR : too many assay type, found : {assay_types_counts.index}")

    # look for empty assay type values
    if T0.loc[T0["assay_type"].isna()].shape[0]:
        exit("ERROR: assay types has missing values")


def sanity_check_assay_sizes(T0, T1):

    t0_assays = T0.input_assay_id.unique()
    t1_assays = T1.input_assay_id.unique()
    num_t0_assay = t0_assays.shape[0]
    num_t1_assay = t1_assays.shape[0]

    if num_t1_assay < num_t0_assay:
        print("WARNING : T1 does not have all input_assay_id present in T0")
        assays_not_in_t1 = T0.loc[~T0["input_assay_id"].isin(t1_assays)]
        print("Assay not in T1:")
        print(assays_not_in_t1)

    if num_t0_assay < num_t1_assay:
        print("ERROR: some input_assay_id present in T1 are not present in T0")
        assays_not_in_t0 = T1.loc[~T1["input_assay_id"].isin(t0_assays)]
        print("Assay not in T0:")
        print(assays_not_in_t0)

    if num_t0_assay != num_t1_assay:
        print(f"ERROR : number input_assay_id differs")
        print(f"WARNING : T0 input_assay_id count: {num_t0_assay}")
        print(f"WARNING : T1 input_assay_id count: {num_t1_assay}")
        exit(
            "Processing will be stopped. Please check for consistency in input_assay_id in T0 and T1."
        )


def sanity_check_compound_sizes(T1, T2):

    t1_compounds = T1.input_compound_id.unique()
    t2_compounds = T2.input_compound_id.unique()
    num_t2_compounds = t2_compounds.shape[0]
    num_t1_compounds = t1_compounds.shape[0]

    if num_t1_compounds < num_t2_compounds:
        print("WARNING : T1 does not have all input_compound_id present in T2")
        compounds_not_in_t1 = T2.loc[~T2["input_compound_id"].isin(t1_compounds)]
        print("Compounds not in T1:")
        print(compounds_not_in_t1)

    if num_t2_compounds < num_t1_compounds:
        print("**** ERROR: some input_compound_id present in T1 are not present in T2")
        compounds_not_in_t2 = T1.loc[~T1["input_compound_id"].isin(t2_compounds)]
        print("Compounds not in T2:")
        print(compounds_not_in_t2)

    if num_t1_compounds != num_t2_compounds:

        print(f"WARNING : T2 input_compound_id count: {num_t2_compounds}")
        print(f"WARNING : T1 input_compound_id count: {num_t1_compounds}")
        print(f"ERROR : number input_compound_id differs!")

        print(
            "Processing will be stopped. Please check for consistency in input_compound_id in T1 and T2."
        )
        exit()


def sanity_check_uniqueness(df, colname, filename):
    # verif input_compound_id duplicates
    duplic = df[colname].duplicated(keep=False)
    if duplic.sum() > 0:
        print(
            f"Found {duplic.sum()} records with *{colname}* present multiple times in {filename}."
        )
        print(df[duplic])

    df_nrows = df.shape[0]
    df_id = df[colname].nunique()

    if df_nrows != df_id:
        exit(
            f"Processing will be stopped. {colname} is not unique. Please check for duplicates in {filename}."
        )


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
