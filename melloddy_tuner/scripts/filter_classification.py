import argparse

import numpy as np
from melloddy_tuner.utils.config import ConfigDict
import time
from pathlib import Path
from typing import Tuple

import pandas as pd
from melloddy_tuner.utils import hash_reference_set
from melloddy_tuner.utils.helper import (
    create_log_files,
    load_config,
    load_key,
    make_dir,
    make_results_dir,
    read_input_file,
    save_df_as_csv,
)
from melloddy_tuner.utils.formatting import ActivityDataFormatting
from pandas.core.frame import DataFrame


def init_arg_parser():
    """Argparser module to load commandline arguments.

    Returns:
        [Namespace]: Arguments from argparser
    """
    parser = argparse.ArgumentParser(description="Classification data filtering")

    parser.add_argument(
        "-ca",
        "--classification_activity_file",
        type=str,
        help="path of the classification task data T4c",
        required=True,
    )
    parser.add_argument(
        "-cw",
        "--classification_weight_table",
        type=str,
        help="path of the classification task definition and metadata T3c",
        required=True,
    )
    parser.add_argument(
        "-mt",
        "--mapping_table_T5",
        type=str,
        help="path to mapping table T5",
        required=False,
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

    output_dir = make_dir(args, "results_tmp", "classification", overwriting)
    create_log_files(output_dir)
    return output_dir


def filter_on_quorum(df: pd.DataFrame, training_quorum: dict, evaluation_quorum: dict):
    for quorum_dict, quorum_name in zip(
        [training_quorum, training_quorum, evaluation_quorum, evaluation_quorum],
        [
            "num_active_total",
            "num_inactive_total",
            "num_active_fold_min",
            "num_inactive_fold_min",
        ],
    ):
        quorum_dict_reduced = {k: v[quorum_name] for k, v in quorum_dict.items()}
        df[f"quorum_{quorum_name}"] = df["assay_type"].map(quorum_dict_reduced)

    df[f"training_quorum_OK"] = (
        df["num_total_actives"] >= df["quorum_num_active_total"]
    ) & (df["num_total_inactives"] >= df[f"quorum_num_inactive_total"])
    df[f"evaluation_quorum_OK"] = (
        df["num_fold_min_actives"] >= df[f"quorum_num_active_fold_min"]
    ) & (df["num_fold_min_inactives"] >= df[f"quorum_num_inactive_fold_min"])
    return df.drop([c for c in df.columns if c.startswith("quorum_")], axis=1)


def map_2_cont_id(data: pd.DataFrame, column_name: str):
    map_id = {val: ind for ind, val in enumerate(np.unique(data[column_name]))}
    map_id_df = pd.DataFrame.from_dict(map_id, orient="index").reset_index()
    map_id_df = map_id_df.rename(
        columns={"index": column_name, 0: "cont_" + column_name}
    )
    data_remapped = pd.merge(data, map_id_df, how="inner", on=column_name)
    return data_remapped


def filter_clf(T3c, T4c, training_quorum, evaluation_quorum, initial_task_weights):

    T4c["is_active"] = T4c["class_label"] == 1
    T4c["is_inactive"] = T4c["class_label"] == -1
    df_counts = (
        T4c.groupby(["classification_task_id", "fold_id"])
        .agg({"is_active": ["sum"], "is_inactive": ["sum"]})
        .groupby(["classification_task_id"])
        .agg(["sum", "min"])
    )
    df_counts.columns = df_counts.columns.map("_".join)
    df_counts = df_counts.rename(
        columns={
            ("is_active_sum_sum"): "num_total_actives",
            ("is_active_sum_min"): "num_fold_min_actives",
            ("is_inactive_sum_sum"): "num_total_inactives",
            ("is_inactive_sum_min"): "num_fold_min_inactives",
        }
    )

    df = T3c.join(df_counts, on="classification_task_id", how="inner")

    df = filter_on_quorum(df, training_quorum, evaluation_quorum)

    # initialize weight columns
    df["aggregation_weight"] = 1
    df.loc[df["is_auxiliary"] | (~df["evaluation_quorum_OK"]), "aggregation_weight"] = 0
    df_tasks = (
        df.loc[df["training_quorum_OK"]]
        .groupby(["input_assay_id"])
        .agg({"classification_task_id": "size"})
        .reset_index()
    )
    df_tasks = df_tasks.rename(columns={("classification_task_id"): "n_tasks"})
    df = pd.merge(
        df, df_tasks, left_on="input_assay_id", right_on="input_assay_id", how="left"
    )

    # set task weight as intital task weight, will leave np.nan for assay types without defined weight
    df.loc[df["training_quorum_OK"], "weight"] = df.loc[
        df["training_quorum_OK"], "assay_type"
    ].map(initial_task_weights)
    # for assay types without an defined initial weight, fill in 1.0 as default value
    # this will cover the non auxiliary assay types
    df.loc[df["training_quorum_OK"], "weight"] = df.loc[
        df["training_quorum_OK"], "weight"
    ].fillna(1.0)
    # now we divide the intial weight by the number of tasks
    df.loc[df["training_quorum_OK"], "weight"] = (
        df.loc[df["training_quorum_OK"], "weight"] / df.n_tasks
    )

    df_training = df[df["training_quorum_OK"]]

    T8c = map_2_cont_id(df_training, "classification_task_id")
    cont_mapping_df = T8c.set_index("classification_task_id")[
        ["cont_classification_task_id"]
    ]

    T8c = df.merge(cont_mapping_df, on="classification_task_id", how="outer")
    T8c = pd.concat(
        [T8c, T3c[~T3c.input_assay_id.isin(T8c.input_assay_id.unique())]],
        ignore_index=True,
    )
    # add missing tasks that are present in T3c

    training_mask = T4c.classification_task_id.isin(
        df_training.classification_task_id.unique()
    )
    T10c, T4c_filtered_out = T4c[training_mask], T4c[~training_mask]
    T10c = T10c.merge(cont_mapping_df, on="classification_task_id", how="outer")
    # descriptor_mask = T5.descriptor_vector_id.isin(
    #     T10c.descriptor_vector_id.unique()
    # )
    # T6 = T5[descriptor_mask]

    # T6_cont = map_2_cont_id(T6, 'descriptor_vector_id')
    # cont_desc_mapping_df = T6_cont.set_index('descriptor_vector_id')[['cont_descriptor_vector_id']]
    # T10c = T10c.merge(
    #     cont_desc_mapping_df, on='descriptor_vector_id', how='outer')

    # Dereplicate
    ind_dup = T10c.index[
        T10c.duplicated(
            subset=["cont_classification_task_id", "descriptor_vector_id"], keep=False
        )
    ]
    T4c_dedup = T10c.loc[ind_dup, :].copy()
    T10c = T10c.drop_duplicates(
        subset=["cont_classification_task_id", "descriptor_vector_id"]
    )

    return T10c, T8c, T4c_filtered_out, T4c_dedup


def write_tmp_output(
    out_dir: Path,
    T10c: pd.DataFrame,
    T8c: pd.DataFrame,
    T4c_filtered_out: pd.DataFrame,
    T4c_dedup: pd.DataFrame,
) -> None:
    """Save csv files of aggregated activity values and data outside the credibililty range.

    Args:
        out_dir (Path): output Path object
        T4c_filtered_out (DataFrame): dataframe containing regression activity data filtered out in training
        T4c_dedup (DataFrame): dataframe containing duplicated regression activity data
    """

    save_df_as_csv(
        out_dir,
        T4c_filtered_out,
        "filtered_out_T4c",
        [
            "classification_task_id",
            "input_assay_id",
            "descriptor_vector_id",
            "fold_id",
            "standard_qualifier",
            "standard_value",
            "threshold",
            "class_label",
        ],
    )
    save_df_as_csv(
        out_dir,
        T4c_dedup,
        "duplicates_T4c",
        [
            "classification_task_id",
            "input_assay_id",
            "descriptor_vector_id",
            "fold_id",
            "standard_qualifier",
            "standard_value",
            "threshold",
            "class_label",
        ],
    )
    save_df_as_csv(
        out_dir,
        T10c,
        "T10c",
        [
            "cont_classification_task_id",
            "descriptor_vector_id",
            "fold_id",
            "class_label",
        ],
    )
    save_df_as_csv(
        out_dir,
        T8c,
        "T8c",
        [
            "cont_classification_task_id",
            "classification_task_id",
            "input_assay_id",
            "assay_type",
            "variance_quorum_OK",
            "is_auxiliary",
            "use_in_regression",
            "threshold",
            "threshold_method",
            "direction",
            "training_quorum_OK",
            "evaluation_quorum_OK",
            "aggregation_weight",
            "weight",
            "num_total_actives",
            "num_fold_min_actives",
            "num_total_inactives",
            "num_fold_min_inactives",
            "n_tasks",
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
    print("Start classification filtering.")
    output_dir = prepare(args, overwriting)
    T3c = read_input_file(args["classification_weight_table"])
    T4c = read_input_file(args["classification_activity_file"])
    T10c, T8c, T4c_filtered_out, T4c_dedup = filter_clf(
        T3c,
        T4c,
        ConfigDict.get_parameters()["training_quorum"]["classification"],
        ConfigDict.get_parameters()["evaluation_quorum"]["classification"],
        ConfigDict.get_parameters()["initial_task_weights"],
    )
    write_tmp_output(output_dir, T10c, T8c, T4c_filtered_out, T4c_dedup)
    print(f"Classification filtering took {time.time() - start:.08} seconds.")
    print(f"Classification filtering done.")


if __name__ == "__main__":
    args = vars(init_arg_parser())
    main(args)
