import os
import json
import argparse
import logging
from pathlib import Path
import time
from typing import Tuple

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
    save_df_as_csv,
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
        mask = np.array([i for i in range(len(qualifiers)) if qualifiers[i] != "<"])
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
    if x["assay_type"].values[0] == "PANEL" or x["assay_type"].values[0] == "OTHER":
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
        [aggr_value, aggr_qualifier], index=["standard_value", "standard_qualifier"]
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
        for assay_type in ["ADME", "PANEL", "OTHER", "AUX_HTS"]:
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


def filter_by_std(df: pd.DataFrame, conf: dict) -> Tuple:
    """Removes tasks that have small std at least in one fold"""
    stds = (
        df[["input_assay_id", "fold_id", "standard_value"]]
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

    return df, df_failed


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
    df = T0.merge(
        T1.merge(T5, on="input_compound_id", how="outer"),
        on="input_assay_id",
        how="outer",
    )
    df.reset_index(drop=True, inplace=True)
    end_time = time.time()
    # Time taken in seconds
    time_taken = end_time - start_time
    print(f"Merge took {time_taken:.08} seconds.")
    # filter values outside the credibility range
    start_time = time.time()
    df, df_failed_range = filter_credibility_range(df, conf)
    end_time = time.time()
    # Time taken in seconds
    time_taken = end_time - start_time
    print(f"Filter credibility range took {time_taken:.08} seconds.")

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
                        aggregate_for_one_task, list(df_dup.groupby("input_assay_id"))
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
    df_aggr, df_failed_std = filter_by_std(df_aggr, conf)
    end_time = time.time()
    time_taken = end_time - start_time
    print(f"Filtering based on standard deviation took {time_taken:.08} seconds.")
    # Add variance_quorum_OK column to T0
    T0_upd = T0.copy()
    T0_upd.loc[:, "variance_quorum_OK"] = False
    ind = T0_upd[T0_upd.input_assay_id.isin(df_aggr.input_assay_id.unique())].index
    T0_upd.loc[ind, "variance_quorum_OK"] = True

    # round to 2 digits
    dose_response_assays = set(
        T0[T0.assay_type.isin(["OTHER", "PANEL"])].input_assay_id.unique()
    )
    ind_to_round = df_aggr[df_aggr.input_assay_id.isin(dose_response_assays)].index
    values = df_aggr.loc[ind_to_round, "standard_value"].values
    values_converted = np.round(values, 2)
    df_aggr.loc[ind_to_round, "standard_value"] = values_converted

    return df_aggr, df_failed_range, df_failed_aggr, df_failed_std, df_dup, T0_upd


def write_tmp_output(
    out_dir: Path,
    df: pd.DataFrame,
    df_failed_range: pd.DataFrame,
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
        ["input_compound_id", "input_assay_id", "standard_qualifier", "standard_value"],
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
    load_config(args)
    load_key(args)
    print("Consistency checks of config and key files.")
    hash_reference_set.main(args)
    print("Start aggregation.")
    output_dir = prepare(args, overwriting)
    T0 = read_input_file(args["assay_file"])
    T1 = read_input_file(args["activity_file"])
    print("Check assay types in T0.")
    sanity_check_assay_type(T0)

    print("Check consistency of input_assay_id between T0 and T1.")
    sanity_check_assay_sizes(T0, T1)
    print("Check uniqueness of T0.")
    sanity_check_uniqueness(T0, colname="input_assay_id", filename=args["assay_file"])
    print(f"Sanity checks took {time.time() - start:.08} seconds.")
    print(f"Sanity checks passed.")

    T5 = read_input_file(args["mapping_table"])
    (
        df_aggr,
        df_failed_range,
        df_failed_aggr,
        df_failed_std,
        df_dup,
        T0_upd,
    ) = aggregate_replicates(
        T0, T1, T5, ConfigDict.get_parameters()["credibility_range"], args["number_cpu"]
    )
    write_tmp_output(
        output_dir,
        df_aggr,
        df_failed_range,
        df_failed_aggr,
        df_failed_std,
        df_dup,
        T0_upd,
    )

    print(f"Replicate aggregation took {time.time() - start:.08} seconds.")
    print(f"Replicate aggregation done.")


if __name__ == "__main__":
    args = vars(init_arg_parser())
    main(args)
