import os
import json
import argparse
import logging
from pathlib import Path
import time
from typing import Tuple
from collections import Counter

import pandas as pd
import numpy as np
import tqdm
from melloddy_tuner.utils import hash_reference_set
from melloddy_tuner.utils.helper import (
    load_config,
    load_key,
    make_dir,
    read_input_file,
    create_log_files,
    sanity_check_assay_sizes,
    sanity_check_assay_type,
    sanity_check_uniqueness,
    sanity_check_binary,
    save_df_as_csv,
    save_run_report,
    validate_T0,
    validate_T1
)
from melloddy_tuner.utils.config import ConfigDict
from multiprocessing import Pool


def init_arg_parser():
    """Argparser module to load commandline arguments.

    Returns:
        [Namespace]: Arguments from argparser
    """
    parser = argparse.ArgumentParser(description="smiles standardization")

    parser.add_argument(
        "-assay",
        "--assay_file",
        type=str,
        help="path of the assay metadata file T0",
        required=True,
    )
    parser.add_argument(
        "-a",
        "--activity_file",
        type=str,
        help="path of the activity data file T1",
        required=True,
    )
    parser.add_argument(
        "-mt",
        "--mapping_table",
        type=str,
        help="path of the mapping table T5",
        required=True,
    )
    parser.add_argument(
        "-c", "--config_file", type=str, help="path of the config file", required=True
    )
    parser.add_argument(
        "-k", "--key_file", type=str, help="path of the key file", required=True
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        help="path to the generated output directory",
        required=True,
    )
    parser.add_argument(
        "-r", "--run_name", type=str, help="name of your current run", required=True
    )
    parser.add_argument(
        "-rh",
        "--ref_hash",
        type=str,
        help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
    )
    parser.add_argument(
        "-ni",
        "--non_interactive",
        help="Enables an non-interactive mode for cluster/server usage",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-cpu", "--number_cpu", type=int, help="number of CPUs", default=1
    )

    args = parser.parse_args()
    return args


def most_common_bin_value(values) -> int:
    """Determines the most common binary value, in case of a tie it returns 1
    Input:
        values - list of binary values, accepted values -1 and 1
    Output:
        int: the most common binary value. In case of a tie it returns 1
    """
    occurence_count = Counter(values)
    print(len(occurence_count.keys()))
    if len(occurence_count.keys()) > 1:
        if occurence_count.most_common(2)[0][1] == occurence_count.most_common(2)[1][1]:
            return 1
        else:
            return occurence_count.most_common(1)[0][0]

    else:
        return occurence_count.most_common(1)[0][0]


def most_common_qualifier(qualifiers: list) -> str:
    """Determines the most common qualifier, in case of a tie including '=' returns '='
    Input:
        qualifiers - list of qualifiers, accepted values '<', '>' and '='
    Output:
        str: the most common qualifier. In case of a tie prefers '='. If a tie is between '<' and '>' - returns None
    """
    counts = []
    for qual in ["<", ">", "="]:
        counts.append((qual, qualifiers.count(qual)))
    counts.sort(key=lambda tup: tup[1], reverse=True)
    if counts[0][1] > counts[1][1]:
        return counts[0][0]
    elif counts[0][0] == "=" or counts[1][0] == "=" or counts[2][1] == counts[0][1]:
        return "="
    else:
        return None


def aggr_median(values, qualifiers) -> Tuple:
    """Identifies median of values and the most common qualifier"""
    return np.median(values), most_common_qualifier(list(qualifiers))


def aggr_min(values, qualifiers) -> Tuple:
    """Identifies the minimum value and teh corresponding qualifier"""
    return values[np.argmin(values)], qualifiers[np.argmin(values)]


def aggr_max(values, qualifiers) -> Tuple:
    """Identifies the maximum values and the corresponding qualifier
    If '<' qualifier is present, only those elements ae considered
    """
    if (">" in qualifiers) or ("=" in qualifiers):
        mask = np.array(
            [i for i in range(len(qualifiers)) if qualifiers[i] != "<"])
        ind = mask[np.argmax(np.array(values)[mask])]
        aggr_value = values[ind]
        aggr_qualifier = qualifiers[ind]
    else:
        aggr_value = values[np.argmax(values)]
        aggr_qualifier = qualifiers[np.argmax(values)]
    return aggr_value, aggr_qualifier


def aggr_absmax(values, qualifiers) -> Tuple:
    """Identifies the value and the qualifier, corresponding to the max absolute values"""
    return values[np.argmax(np.abs(values))], qualifiers[np.argmax(np.abs(values))]


def aggregate(x: pd.DataFrame) -> pd.Series:
    """Aggregates values within a data frame
    Input:
        x: pandas dataframe containing columns "assay_type" and "direction"
    Returns:
        pd.Series with index 'standard_value' and 'standard_qualifier'
    """
    if x["is_binary"].values[0] == True:
        aggr_value = most_common_bin_value(x["standard_value"].values)
        aggr_qualifier = "="
    elif x["assay_type"].values[0] == "NON-CATALOG-PANEL" or x["assay_type"].values[0] == "OTHER":
        aggr_value, aggr_qualifier = aggr_max(
            x["standard_value"].values, x["standard_qualifier"].values
        )
    elif x["assay_type"].values[0] == "ADME":
        aggr_value, aggr_qualifier = aggr_median(
            x["standard_value"].values, x["standard_qualifier"].values
        )
    elif x["direction"].values[0] == "low":
        aggr_value, aggr_qualifier = aggr_min(
            x["standard_value"].values, x["standard_qualifier"].values
        )
    elif x["direction"].values[0] == "high":
        aggr_value, aggr_qualifier = aggr_max(
            x["standard_value"].values, x["standard_qualifier"].values
        )
    else:
        aggr_value, aggr_qualifier = aggr_absmax(
            x["standard_value"].values, x["standard_qualifier"].values
        )

    return pd.Series(
        [aggr_value, aggr_qualifier], index=[
            "standard_value", "standard_qualifier"]
    )


def prepare(args: dict, overwriting: bool) -> Path:
    """Load config and key file,create output directories and setup log files.

    Args:
        args (dict): argparser dictionary

    Returns:
        Path: output directory path
    """

    output_dir = make_dir(args, "results_tmp", "aggregation", overwriting)
    create_log_files(output_dir)
    return output_dir


def filter_credibility_range(df: pd.DataFrame, conf: dict) -> Tuple:
    """Fiilters data according to the credibility range.

    Args:
        df (DataFrame): activity data file containing columns "input_compound_id", "input_assay_id", "standard_qualifier" and "standard_value"
        conf (optional): configuration dictionary containing credible value range
    Returns:
        Tuple[DataFrame, DataFrame]: dataframe with credible values, dataframe with activity data outside of credible value range
    """

    # Remove NaNs
    ind_to_remove = df[df.standard_value.isnull()].index
    df_failed = df.loc[ind_to_remove, :].copy()
    df.drop(axis=0, index=ind_to_remove, inplace=True)

    # Check value ranges
    if conf:
        ind_to_remove = []
        for assay_type in ["ADME", "NON-CATALOG-PANEL", "OTHER", "AUX_HTS"]:
            if assay_type in conf.keys():
                ind_to_remove += list(
                    df[
                        (df.assay_type == assay_type)
                        & (
                            (df.standard_value < conf[assay_type]["min"])
                            | (df.standard_value > conf[assay_type]["max"])
                        )
                    ].index
                )
        df_failed = df_failed.append(df.loc[ind_to_remove, :].copy())
        df.drop(axis=0, index=ind_to_remove, inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df, df_failed


def filter_binary(df: pd.DataFrame) -> Tuple:
    """Fiilters binary data.

    Args:
        df (DataFrame): activity data file containing columns "input_compound_id", "input_assay_id", "standard_qualifier" and "standard_value"
        conf (optional): configuration dictionary
    Returns:
        Tuple[DataFrame, DataFrame]: dataframe with binary vales, dataframe with activity data with non binary values
    """

    # Remove NaNs
    ind_to_remove = df[df.standard_value.isnull()].index
    df_failed = df.loc[ind_to_remove, :].copy()
    df.drop(axis=0, index=ind_to_remove, inplace=True)

    # Check binary values
    ind_to_remove = []
    ind_to_remove += list(
                    df[
                        (df.is_binary == True)
                        & (
                            (df.standard_value != 1)
                            & (df.standard_value != -1)
                        )
                    ].index)
    df_failed = df_failed.append(df.loc[ind_to_remove, :].copy())
    df.drop(axis=0, index=ind_to_remove, inplace=True)

    df.reset_index(drop=True, inplace=True)

    return df, df_failed

def filter_by_std(df: pd.DataFrame, T0: pd.DataFrame, conf: dict) -> Tuple:
    """Removes tasks that have small std at least in one fold, except for catalog assays"""
    T0_upd = T0.copy()
    T0_upd.loc[:, "variance_quorum_OK"] = True
    non_cat_assays = T0[T0.assay_type !=
                        'CATALOG-PANEL'].input_assay_id.unique()
    stds = (
        df[df.input_assay_id.isin(non_cat_assays)][["input_assay_id",
                                                    "fold_id", "standard_value"]]
        .groupby(["input_assay_id", "fold_id"])
        .std()
        .reset_index()
    )
    stds = stds.pivot(
        columns="fold_id", values="standard_value", index="input_assay_id"
    )
    stds.fillna(0, inplace=True)
    min_stds = stds.min(axis=1)
    tasks_to_remove = min_stds[min_stds <= conf["std"]["min"]].index
    ind_to_remove = df[df.input_assay_id.isin(tasks_to_remove)].index
    df_failed = df.loc[ind_to_remove, :].copy()
    df.drop(axis=0, index=ind_to_remove, inplace=True)
    df.reset_index(drop=True, inplace=True)
    ind = T0_upd[T0_upd.input_assay_id.isin(tasks_to_remove)].index
    T0_upd.loc[ind, "variance_quorum_OK"] = False

    cat_assays = T0[T0.assay_type == 'CATALOG-PANEL'].input_assay_id.unique()
    stds = (
        df[df.input_assay_id.isin(cat_assays)][["input_assay_id",
                                                "fold_id", "standard_value"]]
        .groupby(["input_assay_id", "fold_id"])
        .std()
        .reset_index()
    )
    stds = stds.pivot(
        columns="fold_id", values="standard_value", index="input_assay_id"
    )
    stds.fillna(0, inplace=True)
    min_stds = stds.min(axis=1)
    tasks_to_switch_off_regression = min_stds[min_stds <=
                                              conf["std"]["min"]].index
    ind = T0_upd[T0_upd.input_assay_id.isin(
        tasks_to_switch_off_regression)].index
    #T0_upd.loc[ind, "variance_quorum_OK"] = False
    T0_upd.loc[ind, 'use_in_regression'] = False

    return df, T0_upd, df_failed


def map_qualifiers(df: pd.DataFrame) -> pd.DataFrame:
    """Maps qualifiers ['<', '<=', '<<', '>', '>=', '>>', '=': '=', '~'] into ['<', '>', '=']
    Input:
        df: data frame, assumed to contain column 'standard_qualifier' with accepted qualifiers
    Returns:
        the same data frame with 'standard_qualifier' column mapped
    """
    df.fillna({"standard_qualifier": "="}, inplace=True)
    qualifier_map = {
        "<": "<",
        "<=": "<",
        "<<": "<",
        ">": ">",
        ">=": ">",
        ">>": ">",
        "=": "=",
        "~": "=",
    }
    df.loc[:, "standard_qualifier"].replace(qualifier_map, inplace=True)
    return df


def aggregate_for_one_task(df_tuple):

    df_ag = (
        df_tuple[1]
        .groupby(["descriptor_vector_id", "fold_id"])
        .apply(aggregate)
        .reset_index()
    )
    df_ag.loc[:, "input_assay_id"] = df_tuple[0]
    return df_ag


def aggregate_replicates(
    T0: pd.DataFrame, T1: pd.DataFrame, T5: pd.DataFrame, conf: dict, num_cpu: int
) -> Tuple:
    """Wrapper to aggregate replicates.

    Args:
        T0 (DataFrame): dataframe containing assay metadata including columns "input_assay_id", "assay_type" and "direction"
        T1 (DataFrame): activity data file containing columns "input_compound_id", "input_assay_id", "standard_qualifier" and "standard_value"
        T5 (DataFrame): mapping table containing columns "input_compound_id", "descriptor_vector_id" and "fold_id"
        conf (optional): configuration dictionary containing credible value range
        num_cpu: number of CPUs to use for parallelization
    Returns:
        dataframe with aggregated values (T4r)
        dataframe with activity data outside of credible value range
        dataframe with activity data that couldn't be aggregated (tie between < and > in ADME assay)
        dataframe with duplicated measurements
    """

    start_time = time.time()
    
    assay_not_in_T1 = T0[~T0.input_assay_id.isin(T1["input_assay_id"].unique())]
    print(f"{assay_not_in_T1.input_assay_id.nunique()} unique assays are not having data in T1 and will be skipped.")
    T1_T5 = T1.merge(T5, on="input_compound_id", how="outer")
    df = T0.merge(
        T1_T5,
        on="input_assay_id",
        how="inner",
    )
    
    df.reset_index(drop=True, inplace=True)
    end_time = time.time()
    # Time taken in seconds
    time_taken = end_time - start_time
    print(f"Merge took {time_taken:.08} seconds.")
    # filter values outside the credibility range
    start_time = time.time()
   
    ind_non_binary = df[df.is_binary == False].index
    ind_binary = df[df.is_binary == True].index

    df_filter, df_failed_range = filter_credibility_range(
        df.iloc[ind_non_binary].copy(), conf)

    df_filter_bin, df_failed_binary = filter_binary(
        df.iloc[ind_binary].copy())
    df = pd.concat([df_filter_bin, df_filter])
    end_time = time.time()
    # Time taken in seconds
    time_taken = end_time - start_time
    print(f"Filter credibility range and checking binary values took {time_taken:.08} seconds.")

    # Standardise the qualifiers
    start_time = time.time()
    df = map_qualifiers(df)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Standardize qualifiers range took {time_taken:.08} seconds.")
    # Filter data that failed compound standardization
    df = df[~df.descriptor_vector_id.isnull()]
    df[["fold_id", "descriptor_vector_id"]] = df[
        ["fold_id", "descriptor_vector_id"]
    ].astype(int)
    df.reset_index(drop=True, inplace=True)

    # Identify duplicates
    duplicates = df.duplicated(
        subset=["input_assay_id", "descriptor_vector_id", "fold_id"], keep=False
    )
    ind_dup = df.index[duplicates]
    df_dup = df.loc[ind_dup, :].copy()
    ind_uniq = df.index[~duplicates]
    df_aggr = df.loc[
        ind_uniq,
        [
            "input_assay_id",
            "descriptor_vector_id",
            "fold_id",
            "standard_value",
            "standard_qualifier",
        ],
    ]
    # Aggregate replicates
    start_time = time.time()
    if len(df_dup) > 0:
        with Pool(processes=num_cpu) as pool:
            df_aggr_ = list(
                tqdm.tqdm(
                    pool.imap(
                        aggregate_for_one_task, list(
                            df_dup.groupby("input_assay_id"))
                    ),
                    total=len(df_dup.input_assay_id.unique()),
                )
            )
            pool.close()
            pool.join()
        df_aggr = pd.concat([df_aggr, pd.concat(df_aggr_)])
    df_failed_aggr = df_aggr[df_aggr.standard_qualifier.isnull()].copy()
    df_aggr = df_aggr[~df_aggr.standard_qualifier.isnull()].copy()
    df_aggr.reset_index(drop=True, inplace=True)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Aggregate replicates took {time_taken}")

    # Remove tasks with low std
    start_time = time.time()
    df_aggr, T0_upd, df_failed_std = filter_by_std(df_aggr, T0, conf)
    end_time = time.time()
    time_taken = end_time - start_time
    print(
        f"Filtering based on standard deviation took {time_taken:.08} seconds.")
    # Add variance_quorum_OK column to T0
    #T0_upd = T0.copy()
    #T0_upd.loc[:, "variance_quorum_OK"] = False
    # ind = T0_upd[T0_upd.input_assay_id.isin(
    #    df_aggr.input_assay_id.unique())].index
    #T0_upd.loc[ind, "variance_quorum_OK"] = True

    # round to 2 digits
    dose_response_assays = set(
        T0[T0.assay_type.isin(["OTHER", "NON-CATALOG-PANEL"])
           ].input_assay_id.unique()
    )
    ind_to_round = df_aggr[df_aggr.input_assay_id.isin(
        dose_response_assays)].index
    values = df_aggr.loc[ind_to_round, "standard_value"].values
    values_converted = np.round(values, 2)
    df_aggr.loc[ind_to_round, "standard_value"] = values_converted

    return df_aggr, df_failed_range, df_failed_binary, df_failed_aggr, df_failed_std, df_dup, T0_upd


def write_tmp_output(
    out_dir: Path,
    df: pd.DataFrame,
    df_failed_range: pd.DataFrame,
    df_failed_binary: pd.DataFrame,
    df_failed_aggr: pd.DataFrame,
    df_failed_std: pd.DataFrame,
    df_dup: pd.DataFrame,
    T0_upd: pd.DataFrame,
) -> None:
    """Save csv files of aggregated activity values and data outside the credibililty range.

    Args:
        out_dir (Path): output Path object
        df (DataFrame): dataframe containing aggregated activity data
        T0_upd (DataFrame): dataframe with updated T0 info
        df_failed_range (DataFrame): dataframe containing activity data outside the credibility range
        df_failed_aggr (DataFrame): dataframe containing activity data that failed at aggregation step
        df_failed_std (DataFrame): dataframe containing activity data that failed due to low std per task per fold
    """
    save_df_as_csv(
        out_dir,
        df,
        "T4r",
        [
            "input_assay_id",
            "descriptor_vector_id",
            "fold_id",
            "standard_qualifier",
            "standard_value",
        ],
    )
    save_df_as_csv(out_dir, T0_upd, "T0_upd")
    save_df_as_csv(
        out_dir,
        df_failed_range,
        "failed_range_T1",
        ["input_compound_id", "input_assay_id",
            "standard_qualifier", "standard_value"],
    )
    save_df_as_csv(
        out_dir,
        df_failed_binary,
        "failed_binary_T1",
        ["input_compound_id", "input_assay_id",
            "standard_qualifier", "standard_value"],
    )
    
    save_df_as_csv(
        out_dir,
        df_failed_aggr,
        "failed_aggr_T1",
        [
            "descriptor_vector_id",
            "input_assay_id",
            "standard_qualifier",
            "standard_value",
            "fold_id",
        ],
    )
    save_df_as_csv(
        out_dir,
        df_failed_std,
        "failed_std_T1",
        [
            "descriptor_vector_id",
            "input_assay_id",
            "standard_qualifier",
            "standard_value",
            "fold_id",
        ],
    )
    save_df_as_csv(
        out_dir,
        df_dup,
        "duplicates_T1",
        [
            "input_assay_id",
            "input_compound_id",
            "descriptor_vector_id",
            "fold_id",
            "standard_qualifier",
            "standard_value",
        ],
    )


def main(args):
    """General wrapper function for replicate aggregation.

    Args:
        args (dict): Dictionary of arguments from argparser

    Returns:
        DataFrame: Activity file T4r with aggregated values
    """
    start = time.time()

    if args["non_interactive"] is True:
        overwriting = True
    else:
        overwriting = False
    dict_report = {}
    passed_l = []
    load_config(args)
    load_key(args)
    print("Consistency checks of config and key files.")
    hash_reference_set.main(args)
    dict_report["run_parameters"] = args

    print("Start aggregation.")
    dict_aggr = {}
    output_dir = prepare(args, overwriting)
    T0 = read_input_file(args["assay_file"])
    T1 = read_input_file(args["activity_file"])
    validate_T0(T0)
    validate_T1(T1)
    print("Check assay types in T0.")
    
    passed, dict_assay_type = sanity_check_assay_type(T0)
    passed_l.append(passed)
    dict_aggr["assay_type_check"] = dict_assay_type
    print(T0.loc[T0.is_binary == True].shape)
    passed, dict_binary = sanity_check_binary(T0)
    passed_l.append(passed)
    dict_aggr["binary_check"] = dict_binary
    print("Check consistency of input_assay_id between T0 and T1.")
    passed, dict_assay_sizes = sanity_check_assay_sizes(T0, T1)
    passed_l.append(passed)
    dict_aggr["assay_sizes_check"] = dict_assay_sizes
    print("Check uniqueness of T0.")
    passed, dict_unique = sanity_check_uniqueness(
        T0, colname="input_assay_id", filename=args["assay_file"])
    passed_l.append(passed)
    dict_aggr["uniqueness"] = dict_unique

    print(f"Sanity checks took {time.time() - start:.08} seconds.")
    if False in passed_l:
        dict_report["aggregate_values"] = dict_aggr
        save_run_report(args, dict_report, "aggregate_vales")
        exit("Found error. Please check the report.")
    else:
        print(f"Sanity checks passed.")
    T5 = read_input_file(args["mapping_table"])
    (
        df_aggr,
        df_failed_range,
        df_failed_binary,
        df_failed_aggr,
        df_failed_std,
        df_dup,
        T0_upd,
    ) = aggregate_replicates(
        T0, T1, T5, ConfigDict.get_parameters(
        )["credibility_range"], args["number_cpu"]
    )
    dict_aggr["failed_range"] = df_failed_range.shape[0]
    dict_aggr["failed_binary"] = df_failed_binary.shape[0]
    dict_aggr["failed_aggr"] = df_failed_aggr.shape[0]
    dict_aggr["failed_std"] = df_failed_std.shape[0]
    dict_aggr["duplicates"] = df_dup.shape[0]
    dict_report["aggregate_values"] = dict_aggr
    write_tmp_output(
        output_dir,
        df_aggr,
        df_failed_range,
        df_failed_binary,
        df_failed_aggr,
        df_failed_std,
        df_dup,
        T0_upd,
    )
    run_time = time.time() - start
    dict_report["run_time"] = run_time
    save_run_report(args, dict_report, "aggregate_values")
    print(f"Replicate aggregation took {run_time:.08} seconds.")
    print(f"Replicate aggregation done.")


if __name__ == "__main__":
    args = vars(init_arg_parser())
    main(args)
