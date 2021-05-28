"""
Common code for data preparation for the MELLODDY project.

Part 3: Formatting and filtering activity data

"""

import argparse
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
    """Argparser function

    Returns:
        Namespace: arguments from argparser tool
    """
    parser = argparse.ArgumentParser(description="Formatting activity data")
    parser.add_argument(
        "-a",
        "--activity_file",
        type=str,
        help="path of the activity input file",
        required=True,
    )
    parser.add_argument(
        "-d",
        "--descriptor_file",
        type=str,
        help="path of descriptor file containing input compound id and descriptor features and values (T2_descriptors.csv)",
        required=True,
    )
    parser.add_argument(
        "-f",
        "--fold_file",
        type=str,
        help="path of descriptor file containing input compound id and assigned fold ID (T2_folds.csv)",
        required=True,
    )
    parser.add_argument(
        "-w",
        "--weight_table",
        type=str,
        help="path of the weight table file",
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
        "-n",
        "--number_cpu",
        type=int,
        help="number of CPUs for calculation (default: 2 CPUs)",
        default=2,
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


def prepare(args: dict, overwriting: bool):
    """Load config and key file,create output directories and setup log files.

    Args:
        args (dict): argparser dictionary

    Returns:
        Path: output directory path
    """

    output_dir = make_dir(args, "results_tmp", "activity_formatting", overwriting)
    mapping_table_dir = make_dir(args, "mapping_table", None, overwriting)
    create_log_files(output_dir)
    return output_dir, mapping_table_dir


def load_input_file(path: str):
    """Read mapping tables from given path argument

    Args:
        path_dir (str): path to mapping table subfolder

    Returns:
        Tuple(DataFrame, DataFrame, DataFrame): mapping_table_T5, mapping_table_T6, mapping_table_T10
    """

    T2_file = Path(path)
    if T2_file.is_file() is False:
        print("Given file does not exist.")
        quit()

    T2 = read_input_file(T2_file)

    return T2


def do_actvity_formattting(
    df_activity_data: DataFrame,
    mapping_table_T5: DataFrame,
    mapping_table_T10: DataFrame,
) -> object:
    """
    Wrapper to run activity_formatting and return activity format object.

    Args:
        df_activity_data (DataFrame): input activity dataframe
        mapping_table_T5 (DataFrame): mapping table T5
        mapping_table_T10 (DataFrame): mapping table T10

    Returns:
        Object: Activity format object
    """
    act_data_format = ActivityDataFormatting(
        df_activity_data, mapping_table_T5, mapping_table_T10
    )
    del (df_activity_data, mapping_table_T5, mapping_table_T10)
    act_data_format.run_formatting()
    return act_data_format


def output_tmp_results(act_data_format: object) -> Tuple:
    """
    Get failed, duplicated and excluded dataframes from activity data object.

    Args:
        act_data_format (object): activity data object

    Returns:
        Tuple: Dataframes of failed, duplicated and excluded activity data.
    """
    data_failed = act_data_format.filter_failed_structures()
    data_duplicated_id_pairs = act_data_format.data_duplicates
    data_excluded = act_data_format.select_excluded_data()
    return data_failed, data_duplicated_id_pairs, data_excluded


def output_results(
    act_data_format: object, df_weight_table: DataFrame, mapping_table_T6: DataFrame
) -> Tuple:
    """
    Get result files T10 and T11 from activity data object and weight tables T3 and T6. Update mapping of T3.

    Args:
        act_data_format (object): Activity data object
        df_weight_table (DataFrame): weight table T3
        mapping_table_T6 (DataFrame): mapping table T6

    Returns:
        Tuple: Dataframe  T11, T10 and T3_mapped
    """
    act_data_format.remapping_2_cont_ids()
    df_T11 = act_data_format.make_T11(mapping_table_T6).sort_values(
        "cont_descriptor_vector_id"
    )
    df_T10 = act_data_format.data_remapped.sort_values("cont_classification_task_id")
    df_T3_mapped = act_data_format.map_T3(df_weight_table)
    return df_T11, df_T10, df_T3_mapped


def write_tmp_output(
    out_dir: Path,
    data_failed: DataFrame,
    data_duplicated_id_pairs: DataFrame,
    data_excluded: DataFrame,
) -> None:
    """
    Writes output files to additional results folder

    Args:
        out_dir (Path): Path to additional results folder
        data_failed (DataFrame): Dataframe containing activity data from failed structures
        data_duplicated_id_pairs (DataFrame): Dataframe containing duplicated pairs of ids
        data_excluded (DataFrame): Dataframe containing
    """
    save_df_as_csv(
        out_dir,
        data_failed,
        "T4_failed_structures",
        ["input_compound_id", "classification_task_id", "class_label"],
    )
    save_df_as_csv(
        out_dir,
        data_duplicated_id_pairs,
        "T4_duplicates",
        ["classification_task_id", "descriptor_vector_id", "class_label"],
    )
    save_df_as_csv(
        out_dir,
        data_excluded,
        "T4_excluded_data",
        ["classification_task_id", "descriptor_vector_id", "class_label"],
    )


def write_mappting_tables(out_dir: Path, df: DataFrame) -> None:
    """
    Wrapper to save mapping table as csv file

    Args:
        out_dir (Path): Path to mapping table subfolder
        df (DataFrame): mapping table dataframe T3
    """
    save_df_as_csv(out_dir, df, "T3_mapping")


def write_output(out_dir: Path, df_T11: DataFrame, df_T10: DataFrame) -> None:
    """
    Write results files as csv to main results folder

    Args:
        out_dir (Path): Path to main results folder.
        df_T11 (DataFrame): Dataframe T11
        df_T10 (DataFrame): Dataframe T10
    """
    save_df_as_csv(
        out_dir,
        df_T11,
        "T11",
        [
            "cont_descriptor_vector_id",
            "descriptor_vector_id",
            "fp_json",
            "fp_val_json",
            "fold_id",
        ],
    )
    save_df_as_csv(
        out_dir,
        df_T10,
        "T10",
        [
            "cont_descriptor_vector_id",
            "cont_classification_task_id",
            "class_label",
            "fold_id",
        ],
    )


def main(args: dict = None):
    """
    Main function reading input files, executing functions and writing output files.
    """
    start = time.time()
    if args is None:
        args = vars(init_arg_parser())

    if args["non_interactive"] is True:
        overwriting = True
    else:
        overwriting = False

    load_config(args)
    load_key(args)
    print("Consistency checks of config and key files.")
    hash_reference_set.main(args)

    print("Start activity data formatting.")
    output_dir, mapping_table_dir = prepare(args, overwriting)
    results_dir = make_results_dir(args, overwriting)
    # read input files (mapping table T5, T10) activity data T4, and weight table T3
    df_activity_data = read_input_file(args["activity_file"])

    df_weight_table = read_input_file(args["weight_table"])
    mapping_table_T5, mapping_table_T6, mapping_table_T10 = load_mapping_tables(
        args["dir_mapping_tables"]
    )

    # read input files (mapping table T5, T10) activity data T4, and weight table T3
    pd.options.mode.chained_assignment = "raise"

    df_activity_data_formatted = do_actvity_formattting(
        df_activity_data, mapping_table_T5, mapping_table_T10
    )

    data_failed, data_duplicated_id_pairs, data_excluded = output_tmp_results(
        df_activity_data_formatted
    )
    write_tmp_output(output_dir, data_failed, data_duplicated_id_pairs, data_excluded)
    del (data_failed, data_duplicated_id_pairs, data_excluded)

    df_T11, df_T10, df_T3_mapped = output_results(
        df_activity_data_formatted, df_weight_table, mapping_table_T6
    )
    write_mappting_tables(mapping_table_dir, df_T3_mapped)
    write_output(results_dir, df_T11, df_T10)
    del (df_activity_data_formatted, df_T11, df_T10, df_T3_mapped)
    end = time.time()
    print(f"Formatting of activity data took {end - start:.08} seconds.")
    print(f"Activity data processing done.")


if __name__ == "__main__":
    main()
