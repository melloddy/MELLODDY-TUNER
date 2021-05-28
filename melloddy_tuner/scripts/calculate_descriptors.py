"""
Common code for data preparation for the MELLODDY project.

 Calculation of descriptors (fingerprints) with RDKit

"""

import argparse
import json
from melloddy_tuner.utils import hash_reference_set
import os
from typing import Tuple

import numpy as np
from melloddy_tuner.utils.descriptor_calculator import DescriptorCalculator
from melloddy_tuner.utils.df_transformer import DfTransformer

from melloddy_tuner.utils.config import ConfigDict, SecretDict
import time
from pathlib import Path


from melloddy_tuner.utils.helper import (
    concat_desc_folds,
    create_log_files,
    load_config,
    load_key,
    make_dir,
    read_input_file,
    save_df_as_csv,
)
import pandas as pd
from pandas import DataFrame

# import hash_reference_set


def init_arg_parser():
    """Argparser function

    Returns:
        Namespace: arguments from argparser tool
    """
    parser = argparse.ArgumentParser(description="Run Fingerprint calculation")
    parser.add_argument(
        "-s",
        "--structure_file",
        type=str,
        help="path of the structure input file",
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
    """Setup run by creating directories and log files.

    Args:
        args (dict): argparser arguments
        overwriting (bool): overwriting flag

    Returns:
        Tuple(DataFrame, DataFrame): Path to output and mapping_table subdirectories.
    """

    output_dir = make_dir(args, "results_tmp", "descriptors", overwriting)
    create_log_files(output_dir)
    load_config(args)
    load_key(args)
    method_params = ConfigDict.get_parameters()["fingerprint"]
    key = SecretDict.get_secrets()["key"]
    dc = DescriptorCalculator.from_param_dict(
        secret=key, method_param_dict=method_params, verbosity=0
    )
    outcols = ["fp_feat", "fp_val", "success", "error_message"]
    out_types = ["object", "object", "bool", "object"]
    dt = DfTransformer(
        dc,
        input_columns={"canonical_smiles": "smiles"},
        output_columns=outcols,
        output_types=out_types,
        success_column="success",
        nproc=args["number_cpu"],
        verbosity=0,
    )
    return output_dir, dt


def run(df, dt):
    """General warpper function to claculate fingerprints and

    Args:
        df (DataFrame): Dataframe with standardized smiles

    Returns:
        Tuple (DataFrame, DataFrame): a datframe with successfully calculated fold information, datafarem with failed molecules
    """
    return dt.process_dataframe(df)


def main(args: dict = None):
    """Main wrapper to execute descriptor calculation and fold assignment.

    Args:
        args (dict): argparser dict containing relevant
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
    output_dir, dt = prepare(args, overwriting)

    print("Start calculating descriptors.")

    input_file = args["structure_file"]
    output_file = os.path.join(output_dir, "T2_descriptors.csv")
    error_file = os.path.join(output_dir, "T2_descriptors.FAILED.csv")
    # dupl_file = os.path.join(output_dir, "T2_descriptors.DUPLICATES.csv")
    # mapping_file_T5 = os.path.join(mapping_table_dir, "T5.csv")
    # mapping_file_T6 = os.path.join(mapping_table_dir, "T6.csv")

    df = pd.read_csv(input_file)
    df_processed, df_failed = dt.process_dataframe(df)
    df_processed.to_csv(output_file, index=False)
    df_failed.to_csv(error_file, index=False)

    # df_T5, df_T6, df_duplicates = format_dataframe(df_processed)
    # df_duplicates.to_csv(dupl_file, index=False)
    # df_T5.to_csv(mapping_file_T5, index=False)
    # df_T6.to_csv(mapping_file_T6, index=False)
    end = time.time()
    print(f"Fingerprint calculation took {end - start:.08} seconds.")
    print(f"Descriptor calculation done.")


if __name__ == "__main__":
    main()
