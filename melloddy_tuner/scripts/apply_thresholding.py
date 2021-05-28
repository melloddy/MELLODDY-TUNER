import itertools
import os
import json
import argparse
import logging
from pathlib import Path
import time
from typing import Tuple

import pandas as pd
import numpy as np
import math

from pandas.core.frame import DataFrame
from tqdm.std import tqdm

from melloddy_tuner.utils import hash_reference_set
from melloddy_tuner.utils.helper import (
    load_config,
    load_key,
    make_dir,
    read_input_file,
    create_log_files,
    save_df_as_csv,
)
from melloddy_tuner.utils.config import ConfigDict
from multiprocessing import Pool
import tqdm
import multiprocessing

multiprocessing.set_start_method("fork", force=True)


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
        help="path of the activity data file T4r",
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


def prepare(args: dict, overwriting: bool) -> Path:
    """Load config and key file,create output directories and setup log files.

    Args:
        args (dict): argparser dictionary

    Returns:
        Path: output directory path
    """

    output_dir = make_dir(args, "results_tmp", "thresholding", overwriting)
    create_log_files(output_dir)
    return output_dir


def convert_qualified_values(df_assay):

    prefix = "<"
    l_index = df_assay[df_assay["standard_qualifier"] == prefix].index
    df_assay.loc[l_index, "standard_value"] -= np.log10(2.0)

    prefix = ">"
    l_index = df_assay[df_assay["standard_qualifier"] == prefix].index
    df_assay.loc[l_index, "standard_value"] += np.log10(2.0)

    values = df_assay["standard_value"].values

    return values


def get_class_label(values, qualifiers, threshold, direction):

    class_label = np.zeros(values.shape, dtype=int)

    if direction == "high":
        # active values
        condition_active = np.logical_and(values >= threshold, qualifiers != "<")
        class_label = np.where(condition_active, 1, class_label)
        # inactive values
        condition_inactive = np.logical_and(values < threshold, qualifiers != ">")
        class_label = np.where(condition_inactive, -1, class_label)
        condition_inactive = np.logical_and(values == threshold, qualifiers == "<")
        class_label = np.where(condition_inactive, -1, class_label)
        # ambiguous values
        condition_missing = np.logical_and(values > threshold, qualifiers == "<")
        class_label = np.where(condition_missing, np.nan, class_label)
        condition_missing = np.logical_and(values < threshold, qualifiers == ">")
        class_label = np.where(condition_missing, np.nan, class_label)
    elif direction == "low":
        # active values
        condition_active = np.logical_and(values <= threshold, qualifiers != ">")
        class_label = np.where(condition_active, 1, class_label)
        # inactive values
        condition_inactive = np.logical_and(values > threshold, qualifiers != "<")
        class_label = np.where(condition_inactive, -1, class_label)
        condition_inactive = np.logical_and(values == threshold, qualifiers == ">")
        class_label = np.where(condition_inactive, -1, class_label)
        # ambiguous values
        condition_missing = np.logical_and(values < threshold, qualifiers == ">")
        class_label = np.where(condition_missing, np.nan, class_label)
        condition_missing = np.logical_and(values > threshold, qualifiers == "<")
        class_label = np.where(condition_missing, np.nan, class_label)

    return class_label


def get_class_label_HTS(values, qualifiers, threshold, direction):

    if direction == "high":
        condition_active = values >= threshold
    elif direction == "low":
        threshold = -1.0 * threshold
        condition_active = values <= threshold
    else:
        condition_active = np.abs(values) >= threshold

    class_label = np.where(condition_active, 1, -1)

    return class_label, threshold


def get_thresholds_dose_response(df_assay, quorum_num_active, quorum_num_inactive):
    l_thresh = []
    n_actives = 0
    n_inactives = 0
    percentage_actives = 0
    thresh_value = 0
    # collect expert thresholds
    columns_expert_threshold = [
        "expert_threshold_1",
        "expert_threshold_2",
        "expert_threshold_3",
        "expert_threshold_4",
        "expert_threshold_5",
    ]
    l_thresh_expert = [df_assay[column].iloc[0] for column in columns_expert_threshold]
    l_thresh_expert = [i for i in l_thresh_expert if not math.isnan(i)]

    if len(l_thresh_expert) > 3:
        # use expert thresholds only
        l_thresh_expert = [(i, "expert") for i in l_thresh_expert]
        l_thresh = l_thresh_expert
    elif len(l_thresh_expert) in [1, 2, 3]:
        # add auxiliary sandwich
        thresh_expert_min_value = np.min(l_thresh_expert)
        thresh_expert_max_value = np.max(l_thresh_expert)
        l_thresh_expert = [(i, "expert") for i in l_thresh_expert]
        l_thresh = l_thresh_expert
        aux_low = thresh_expert_min_value - 0.5
        aux_high = thresh_expert_max_value + 0.5
        l_thresh.append((aux_low, "aux_low"))
        l_thresh.append((aux_high, "aux_high"))
    elif len(l_thresh_expert) == 0:
        # compute default target threshold (fixed-adaptive)
        l_thresh = []
        values = df_assay["standard_value"].values
        qualifiers = df_assay["standard_qualifier"].values
        direction = "high"
        l_trial_threshold = [8.0, 7.0, 6.0, 5.0, 4.7, 4.4]
        for thresh_value in l_trial_threshold:
            # determine actives/inactives index
            class_label = get_class_label(values, qualifiers, thresh_value, direction)
            n_actives = len(class_label[class_label == 1])
            n_inactives = len(class_label[class_label == -1])
            if (n_actives + n_inactives) != 0:
                percentage_actives = n_actives / (n_actives + n_inactives)
                if (
                    percentage_actives >= 0.2
                    and n_actives >= quorum_num_active
                    and n_inactives >= quorum_num_inactive
                ):
                    break

        if (
            percentage_actives >= 0.2
            and n_actives >= quorum_num_active
            and n_inactives >= quorum_num_inactive
        ):
            thresh = (thresh_value, "fixed_adaptive")
        else:
            # use median as fallback option
            df_assay_tmp = df_assay.copy()
            values = convert_qualified_values(df_assay_tmp)
            thresh_value = np.quantile(values, 0.5, interpolation="linear")
            thresh = (thresh_value, "median")

        l_thresh.append(thresh)

        # add auxiliary sandwich
        aux_low = thresh_value - 0.5
        aux_high = thresh_value + 0.5
        l_thresh.append((aux_low, "aux_low"))
        l_thresh.append((aux_high, "aux_high"))

    return l_thresh


def get_thresholds_ADME(df_assay):
    thresh_value_median = 0
    thresh_value_quartile = 0

    # collect expert thresholds
    columns_expert_threshold = [
        "expert_threshold_1",
        "expert_threshold_2",
        "expert_threshold_3",
        "expert_threshold_4",
        "expert_threshold_5",
    ]
    l_thresh_expert = [df_assay[column].iloc[0] for column in columns_expert_threshold]
    l_thresh_expert = [i for i in l_thresh_expert if not math.isnan(i)]
    # add threshold name 'expert'
    l_thresh_expert = [(i, "expert") for i in l_thresh_expert]

    if len(l_thresh_expert) > 0:
        # use expert thresholds
        l_thresh = l_thresh_expert
    else:
        # compute default thresholds
        l_thresh = []

        # select non-qualifier values for computing threshold
        prefix = "="
        l_index = df_assay[df_assay["standard_qualifier"] == prefix].index
        values = df_assay.loc[l_index, "standard_value"].values

        if values.size > 1:
            # compute median threshold
            thresh_value_median = np.quantile(values, 0.5, interpolation="linear")
            l_thresh.append((thresh_value_median, "median"))

            # compute quartile threshold
            direction = df_assay["direction"].iloc[0]
            if direction == "high":
                thresh_value_quartile = np.quantile(
                    values, 0.75, interpolation="linear"
                )
            elif direction == "low":
                thresh_value_quartile = np.quantile(
                    values, 0.25, interpolation="linear"
                )

            # add quartile threshold only if distinct from median
            if thresh_value_quartile != thresh_value_median:
                l_thresh.append((thresh_value_quartile, "quartile"))

    return l_thresh


def apply_thresholding(
    df_assay, assay_type, thresh, direction, columns_T4c, columns_T3c
):

    # compute class label
    threshold = thresh[0]
    threshold_method = thresh[1]
    values = df_assay["standard_value"].values
    qualifiers = df_assay["standard_qualifier"].values

    if assay_type == "AUX_HTS":
        class_label, threshold = get_class_label_HTS(
            values, qualifiers, threshold, direction
        )
    else:
        class_label = get_class_label(values, qualifiers, threshold, direction)

    # build T4c
    df_assay["class_label"] = class_label
    df_assay["threshold"] = threshold
    T4c_assay = df_assay[columns_T4c]

    # build T3c
    tmp_T3c = df_assay.head(1).copy()
    tmp_T3c["threshold_method"] = threshold_method
    tmp_T3c["direction"] = direction

    if tmp_T3c["assay_type"].iloc[0] == "AUX_HTS" or threshold_method in [
        "aux_low",
        "aux_high",
    ]:
        tmp_T3c["is_auxiliary"] = True
    else:
        tmp_T3c["is_auxiliary"] = False

    T3c_assay = tmp_T3c[columns_T3c]

    return T4c_assay, T3c_assay


def calculate_single_assay(df_tuple) -> list:
    # Load HTS threshold if HTS data present
    l_assay_types = df_tuple[1]["assay_type"].unique()
    thresh_HTS = int()
    if "AUX_HTS" in l_assay_types:
        thresh_HTS = ConfigDict.get_parameters()["global_thresholds"]["AUX_HTS"]

    # Load data quora for fixed-adaptive threshold
    quorum_num_active = ConfigDict.get_parameters()["training_quorum"][
        "classification"
    ]["OTHER"]["num_active_total"]
    quorum_num_inactive = ConfigDict.get_parameters()["training_quorum"][
        "classification"
    ]["OTHER"]["num_inactive_total"]

    # Initialize T3c, T4c dataframes
    columns_T3c = [
        "input_assay_id",
        "assay_type",
        "variance_quorum_OK",
        "use_in_regression",
        "is_auxiliary",
        "threshold",
        "threshold_method",
        "direction",
    ]
    columns_T4c = [
        "descriptor_vector_id",
        "fold_id",
        "input_assay_id",
        "standard_qualifier",
        "standard_value",
        "threshold",
        "class_label",
    ]
    df_T3c = pd.DataFrame(columns=columns_T3c)
    df_T4c = pd.DataFrame(columns=columns_T4c)

    tmp_assay = df_tuple[1]
    assay_type = tmp_assay["assay_type"].iloc[0]
    l_thresh = []
    if assay_type in ["OTHER", "PANEL"]:
        l_thresh = get_thresholds_dose_response(
            tmp_assay, quorum_num_active, quorum_num_inactive
        )
    elif assay_type == "ADME":
        l_thresh = get_thresholds_ADME(tmp_assay)
    elif assay_type == "AUX_HTS":
        l_thresh = [(thresh_HTS, "fixed")]
    for thresh in l_thresh:

        # Generate new assay instance
        tmp_assay.loc[:, "input_assay_id"] = df_tuple[0]
        df_assay = tmp_assay.copy()
        # Convert standard_value to class_label
        direction = df_assay["direction"].iloc[0]
        if assay_type == "AUX_HTS":
            T4c_assay, T3c_assay = apply_thresholding(
                df_assay, assay_type, thresh, direction, columns_T4c, columns_T3c
            )
        else:
            if direction != "low":
                direction = "high"
            T4c_assay, T3c_assay = apply_thresholding(
                df_assay, assay_type, thresh, direction, columns_T4c, columns_T3c
            )

        df_T4c = df_T4c.append(T4c_assay)
        df_T3c = df_T3c.append(T3c_assay)

    return df_T4c, df_T3c


def run(T0: DataFrame, T4r: DataFrame, num_cpu: int) -> Tuple:
    """
    Execute thresholding in parallel

    Args:
        T0 (DataFrame): input dataframe T0
        T4r (DataFrame): input dataframe T4r
        num_cpu (int): Number of CPU cores to use

    Returns:
        Tuple: T4c, T3c
    """
    T0_failed = T0[T0.variance_quorum_OK == False].copy()
    T0_passed = T0[T0.variance_quorum_OK == True].copy()
    df = T0_passed.merge(T4r, on="input_assay_id", how="left")
    l_assay_ids = df["input_assay_id"].unique()
    with Pool(processes=num_cpu) as pool:
        df_thres = list(
            tqdm.tqdm(
                pool.imap(calculate_single_assay, df.groupby("input_assay_id")),
                total=len(l_assay_ids),
            )
        )
        pool.close()
        pool.join()
    df_T4c, df_T3c = zip(*df_thres)
    df_T4c = pd.concat(df_T4c)
    df_T3c = pd.concat(df_T3c)
    df_T3c = pd.concat([df_T3c, T0_failed], ignore_index=True)
    df_T3c.sort_values("input_assay_id", inplace=True)
    df_T4c.sort_values("input_assay_id", inplace=True)
    df_T3c["classification_task_id"] = (
        df_T3c.groupby(["input_assay_id", "threshold"])
        .ngroup()
        .replace(-1, np.nan)
        .add(1)
    )
    df_T4c = df_T4c.merge(
        df_T3c[["input_assay_id", "classification_task_id", "threshold"]],
        on=["input_assay_id", "threshold"],
        how="left",
    )
    return df_T4c, df_T3c


def write_failed_output(
    out_dir: Path, df_T4c_failed: pd.DataFrame, columns_T4c: list
) -> None:
    """Save csv files of activity data with ambiguous class labels.

    Args:
        out_dir (Path): output Path object
        df_T4c (DataFrame): dataframe containing classified activity data
    """
    save_df_as_csv(out_dir, df_T4c_failed, "T4c.FAILED", columns_T4c)


def write_tmp_output(
    out_dir: Path,
    df_T4c: pd.DataFrame,
    df_T3c: pd.DataFrame,
    columns_T4c: list,
    columns_T3c: list,
) -> None:
    """Save csv files of classified activity data.

    Args:
        out_dir (Path): output Path object
        df_T4c (DataFrame): dataframe containing classified activity data
        df_T3c (DataFrame): dataframe containing classification threshold definitions
    """
    save_df_as_csv(out_dir, df_T4c, "T4c", columns_T4c)
    save_df_as_csv(out_dir, df_T3c, "T3c", columns_T3c)


def main(args):
    """General wrapper function for thresholding.

    Args:
        args (dict): Dictionary of arguments from argparser

    Returns:
        df_T4c (DataFrame): dataframe containing classified activity data
        df_T3c (DataFrame): dataframe containing classification threshold definitions
    """
    start = time.time()

    if args["non_interactive"] is True:
        overwriting = True
    else:
        overwriting = False
    load_config(args)
    load_key(args)
    num_cpu = args["number_cpu"]
    print("Consistency checks of config and key files.")
    hash_reference_set.main(args)
    print("Start thresholding.")

    # Load files
    output_dir = prepare(args, overwriting)
    T0 = read_input_file(args["assay_file"])
    # T0 = T0.astype({'input_assay_id': 'str'})
    T4r = read_input_file(args["activity_file"])
    # T4r = T4r.astype({'input_assay_id': 'str'})
    # Merge T0 and T4r on input_assay_id

    df_T4c, df_T3c = run(T0, T4r, num_cpu)

    # Write final dataframes (T4c, T3c)
    columns_T3c = [
        "classification_task_id",
        "input_assay_id",
        "assay_type",
        "variance_quorum_OK",
        "use_in_regression",
        "is_auxiliary",
        "threshold",
        "threshold_method",
        "direction",
    ]
    columns_T4c = [
        "classification_task_id",
        "descriptor_vector_id",
        "fold_id",
        "input_assay_id",
        "standard_qualifier",
        "standard_value",
        "threshold",
        "class_label",
    ]

    df_T4c.sort_values("classification_task_id", inplace=True)
    df_T3c.sort_values("classification_task_id", inplace=True)

    # Filter ambiguous class labels
    df_T4c_failed = df_T4c[df_T4c.class_label.isna()]
    df_T4c = df_T4c[~df_T4c.class_label.isna()]

    write_failed_output(output_dir, df_T4c_failed, columns_T4c)
    write_tmp_output(output_dir, df_T4c, df_T3c, columns_T4c, columns_T3c)

    print(f"Thresholding took {time.time() - start:.08} seconds.")
    print(f"Thresholding done.")


if __name__ == "__main__":
    args = vars(init_arg_parser())
    main(args)
