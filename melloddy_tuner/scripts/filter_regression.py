import argparse
from pathlib import Path
import time
from typing import Tuple

import pandas as pd
import numpy as np

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


def init_arg_parser():
    """Argparser module to load commandline arguments.

    Returns:
        [Namespace]: Arguments from argparser
    """
    parser = argparse.ArgumentParser(description="smiles standardization")

    parser.add_argument(
        "-ra",
        "--regression_activity_file",
        type=str,
        help="path of the (censored) regression task data T4r",
        required=True,
    )
    parser.add_argument(
        "-rw",
        "--regression_weight_table",
        type=str,
        help="path of the (censored) regression task definition and metadata T3r",
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

    args = parser.parse_args()
    return args


def prepare(args: dict, overwriting: bool) -> Path:
    """Load config and key file,create output directories and setup log files.

    Args:
        args (dict): argparser dictionary

    Returns:
        Path: output directory path
    """

    output_dir = make_dir(args, "results_tmp", "regression", overwriting)
    create_log_files(output_dir)
    return output_dir


def filter_regression_tasks(
    T0: pd.DataFrame,
    T4r: pd.DataFrame,
    training_quorum: dict,
    evaluation_quorum: dict,
    initial_task_weights: dict,
    censored_downweighting: dict,
) -> Tuple:
    """Wrapper to aggregate replicates.

    Args:
        T0 (DataFrame): dataframe containing assay metadata including columns "input_assay_id", "assay_type" and "direction"
        T4r (DataFrame): activity data file containing columns "input_compound_id", "input_assay_id", "standard_qualifier" and "standard_value"
        training_quorum (dict): configuration dictionary containing the training quorum
        evaluation_quorum (dict): configuration dictionary containing the evaluation quorum
        initial_task_weights (dict): configuration dictionary containing the initial task weights
        censored_downweighting (dict): configuration dictionary containing the downweighting config schema
    Returns:
        dataframe with regression task data T10r
        dataframe with regression tasks definition and metadata T8r
        dataframe with regression task data filtered out for training
        dataframe with duplicated regression task data
    """
    T3r = T0.copy()
    if T3r["assay_type"].iloc[0] == "AUX_HTS":
        T3r["is_auxiliary"] = True
    else:
        T3r["is_auxiliary"] = False
    # Need to create columns for training_quorum_OK and evaluation_quorum_OK
    T4r["is_uncensored"] = T4r["standard_qualifier"] == "="
    df_counts = (
        T4r.groupby(["input_assay_id", "fold_id"])
        .is_uncensored.agg(["count", "sum"])
        .groupby("input_assay_id")
        .agg(["sum", "min"])
    )
    df_counts.columns = df_counts.columns.map("_".join)
    df_counts = df_counts.rename(
        columns={
            ("count_sum"): "num_total",
            ("sum_sum"): "num_uncensored_total",
            ("count_min"): "num_fold_min",
            ("sum_min"): "num_uncensored_fold_min",
        }
    )
    df_counts["num_censored_total"] = (
        df_counts["num_total"] - df_counts["num_uncensored_total"]
    )
    df = T3r.join(df_counts, on="input_assay_id", how="inner")

    df = filter_on_quorum(df, training_quorum, evaluation_quorum)

    # initialize weight columns
    df["aggregation_weight"] = 1
    df.loc[df["is_auxiliary"] | (~df["evaluation_quorum_OK"]), "aggregation_weight"] = 0

    df["weight"] = 1
    df.loc[df["is_auxiliary"], "weight"] = df[df["is_auxiliary"]]["assay_type"].map(
        initial_task_weights
    )

    # TODO: calculatre fraction censored
    fraction_censored = df["num_censored_total"] / df["num_total"]
    # clauclation censored_weight, pass the the censored_downweighting disctionary as kayword arguments
    df["censored_weight"] = censored_weight_transformation(
        fraction_censored, **censored_downweighting
    )

    # Assign continuous_regression_task_id and reindex
    df_training = df[df["training_quorum_OK"] & df["use_in_regression"]].copy()
    df_training.loc[:, "regression_task_id"] = df_training["input_assay_id"]
    T8r = map_2_cont_id(df_training, "regression_task_id")
    cont_mapping_df = T8r.set_index("input_assay_id")[["cont_regression_task_id"]]
    T8r = df.merge(cont_mapping_df, on="input_assay_id", how="outer")
    T8r = pd.concat(
        [T8r, T3r[~T3r.input_assay_id.isin(T8r.input_assay_id.unique())]],
        ignore_index=True,
    )

    training_mask = T4r.input_assay_id.isin(df_training.input_assay_id.unique())
    T10r, T4r_filtered_out = T4r[training_mask], T4r[~training_mask]
    T10r = T10r.merge(cont_mapping_df, on="input_assay_id", how="outer")

    # Dereplicate
    ind_dup = T10r.index[
        T10r.duplicated(
            subset=["cont_regression_task_id", "descriptor_vector_id"], keep=False
        )
    ]
    T4r_dedup = T10r.loc[ind_dup, :].copy()
    T10r = T10r.drop_duplicates(
        subset=["cont_regression_task_id", "descriptor_vector_id"]
    )

    return T10r, T8r, T4r_filtered_out, T4r_dedup


def censored_weight_transformation(
    fraction_censored: pd.Series, knock_in_barrier: float
):
    if (knock_in_barrier > 1.0) or (knock_in_barrier < 0.0):
        raise ValueError(
            "knock_in_barier value {0} is oustode of the allowed range from 0.0 to 1.0".format(
                knock_in_barrier
            )
        )
    # seed the censored weight Series with ones
    censored_weight = pd.Series(
        np.ones(fraction_censored.shape[0], dtype=float), index=fraction_censored.index
    )
    # define amsk for the records needing censored downweighting
    mask = fraction_censored > knock_in_barrier
    # compute censored_weight for values ifd mask == True
    censored_weight.loc[mask] = (
        knock_in_barrier * (1.0 - fraction_censored.loc[mask])
    ) / (fraction_censored.loc[mask] * (1 - knock_in_barrier))
    return censored_weight


def filter_on_quorum(df: pd.DataFrame, training_quorum: dict, evaluation_quorum: dict):
    for quorum_dict, quorum_name in zip(
        [training_quorum, training_quorum, evaluation_quorum, evaluation_quorum],
        [
            "num_total",
            "num_uncensored_total",
            "num_fold_min",
            "num_uncensored_fold_min",
        ],
    ):
        quorum_dict_reduced = {k: v[quorum_name] for k, v in quorum_dict.items()}
        df[f"quorum_{quorum_name}"] = df["assay_type"].map(quorum_dict_reduced)

    df[f"training_quorum_OK"] = (df["num_total"] >= df["quorum_num_total"]) & (
        df["num_uncensored_total"] >= df[f"quorum_num_uncensored_total"]
    )
    df[f"evaluation_quorum_OK"] = (
        df["num_uncensored_fold_min"] >= df[f"quorum_num_uncensored_fold_min"]
    ) & (df["num_uncensored_total"] >= df[f"quorum_num_uncensored_total"])
    return df.drop([c for c in df.columns if c.startswith("quorum_")], axis=1)


def map_2_cont_id(data: pd.DataFrame, column_name: str):
    map_id = {val: ind for ind, val in enumerate(np.unique(data[column_name]))}
    map_id_df = pd.DataFrame.from_dict(map_id, orient="index").reset_index()
    map_id_df = map_id_df.rename(
        columns={"index": column_name, 0: "cont_" + column_name}
    )
    data_remapped = pd.merge(data, map_id_df, how="inner", on=column_name)
    return data_remapped


def write_tmp_output(
    out_dir: Path,
    T10r: pd.DataFrame,
    T8r: pd.DataFrame,
    T4r_filtered_out: pd.DataFrame,
    T4r_dedup: pd.DataFrame,
) -> None:
    """Save csv files of aggregated activity values and data outside the credibililty range.

    Args:
        out_dir (Path): output Path object
        T10r (DataFrame): dataframe containing deduplicated regression task data
        T8r (DataFrame): dataframe containing deduplicated regression task definitions and metadata
        T4r_filtered_out (DataFrame): dataframe containing regression activity data filtered out in training
        T4r_dedup (DataFrame): dataframe containing duplicated regression activity data
    """
    save_df_as_csv(
        out_dir,
        T10r,
        "T10r",
        [
            "input_assay_id",
            "descriptor_vector_id",
            "fold_id",
            "standard_qualifier",
            "standard_value",
            "cont_regression_task_id",
        ],
    )
    save_df_as_csv(
        out_dir,
        T8r,
        "T8r",
        [
            "cont_regression_task_id",
            "input_assay_id",
            "assay_type",
            "variance_quorum_OK",
            "is_auxiliary",
            "use_in_regression",
            "expert_threshold_1",
            "expert_threshold_2",
            "expert_threshold_3",
            "expert_threshold_4",
            "expert_threshold_5",
            "direction",
            "training_quorum_OK",
            "evaluation_quorum_OK",
            "aggregation_weight",
            "weight",
            "censored_weight",
        ],
    )
    save_df_as_csv(
        out_dir,
        T4r_filtered_out,
        "filtered_out_T4r",
        [
            "input_assay_id",
            "descriptor_vector_id",
            "fold_id",
            "standard_qualifier",
            "standard_value",
        ],
    )
    save_df_as_csv(
        out_dir,
        T4r_dedup,
        "duplicates_T4r",
        [
            "input_assay_id",
            "descriptor_vector_id",
            "fold_id",
            "standard_qualifier",
            "standard_value",
            "cont_regression_task_id",
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
    print("Start Regression filtering.")
    output_dir = prepare(args, overwriting)
    T0 = read_input_file(args["regression_weight_table"])
    T4r = read_input_file(args["regression_activity_file"])
    T10r, T8r, T4r_filtered_out, T4r_dedup = filter_regression_tasks(
        T0,
        T4r,
        ConfigDict.get_parameters()["training_quorum"]["regression"],
        ConfigDict.get_parameters()["evaluation_quorum"]["regression"],
        ConfigDict.get_parameters()["initial_task_weights"],
        ConfigDict.get_parameters()["censored_downweighting"],
    )
    write_tmp_output(output_dir, T10r, T8r, T4r_filtered_out, T4r_dedup)

    print(f"Replicate aggregation took {time.time() - start:.08} seconds.")
    print(f"Replicate aggregation done.")


if __name__ == "__main__":
    args = vars(init_arg_parser())
    main(args)
