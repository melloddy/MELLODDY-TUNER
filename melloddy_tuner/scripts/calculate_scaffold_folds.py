from melloddy_tuner.utils import hash_reference_set
import os
import json
import argparse
import logging
from pathlib import Path
import time
from typing import Tuple
import pandas as pd

from pandas import DataFrame


import melloddy_tuner as tuner

# from melloddy_tuner.utils import hash_reference_set
from melloddy_tuner.utils.scaffold_folding import ScaffoldFoldAssign
from melloddy_tuner.utils.df_transformer import DfTransformer
from melloddy_tuner.utils.config import ConfigDict, SecretDict
from melloddy_tuner.utils.helper import (
    format_dataframe,
    load_config,
    load_key,
    make_dir,
    create_log_files,
)


def init_arg_parser():
    """Argparser module to load commandline arguments.

    Returns:
        [Namespace]: Arguments from argparser
    """
    parser = argparse.ArgumentParser(description="smiles standardization")

    parser.add_argument(
        "-s",
        "--structure_file",
        type=str,
        help="path of the structure descriptor file (results_tmp/descriptors/T2_descriptors.csv)",
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
        help="number of CPUs for calculation (default: 1 CPUs)",
        default=1,
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


def prepare(args):
    """
    Prepare output directories and instantiate df tansformer object for scaffold based folding

    Args:
        args (dict): argparser arguments

    Returns:
        Tuple(Path, DfTransformer): Path to output directory and instatitaed DfTranfomer for sccaffold folding


    """
    output_dir = make_dir(args, "results_tmp", "folding", args["non_interactive"])
    mapping_table_dir = make_dir(args, "mapping_table", None, args["non_interactive"])

    create_log_files(output_dir)
    create_log_files(mapping_table_dir)

    load_config(args)
    load_key(args)
    key = SecretDict.get_secrets()["key"]
    method_params = ConfigDict.get_parameters()["scaffold_folding"]
    sa = ScaffoldFoldAssign.from_param_dict(
        secret=key, method_param_dict=method_params, verbosity=0
    )
    outcols = ["murcko_smiles", "sn_smiles", "fold_id", "success", "error_message"]
    out_types = ["object", "object", "int", "bool", "object"]
    dt = DfTransformer(
        sa,
        input_columns={"canonical_smiles": "smiles"},
        output_columns=outcols,
        output_types=out_types,
        success_column="success",
        nproc=args["number_cpu"],
        verbosity=0,
    )
    return output_dir, mapping_table_dir, dt


def run(df, dt):
    """General warpper function to calculate folds by scaffold

    Args:
        df (DataFrame): Dataframe with standardized smiles

    Returns:
        Tuple (DataFrame, DataFrame): a datframe with successfully calculated fold information, datafarem with failed molecules
    """
    return dt.process_dataframe(df)


def main(args: dict = None):
    """General wrapper function to calculate folds from standardized smiles.

    Args:
        args (dict): Dictionary of arguments from argparser

    """
    start = time.time()
    if args is None:
        args = vars(init_arg_parser())
    output_dir, mapping_table_dir, dt = prepare(args)

    hash_reference_set.main(args)

    print("Start computing folds.")

    input_file = args["structure_file"]
    output_file = os.path.join(output_dir, "T2_folds.csv")
    error_file = os.path.join(output_dir, "T2_folds.FAILED.csv")

    dupl_file = os.path.join(output_dir, "T2_descriptor_vector_id.DUPLICATES.csv")
    mapping_file_T5 = os.path.join(mapping_table_dir, "T5.csv")
    mapping_file_T6 = os.path.join(mapping_table_dir, "T6.csv")

    df = pd.read_csv(input_file)
    df_processed, df_failed = dt.process_dataframe(df)
    df_processed.to_csv(output_file, index=False)
    df_failed.to_csv(error_file, index=False)

    df_T5, df_T6, df_duplicates = format_dataframe(df_processed)
    df_duplicates.to_csv(dupl_file, index=False)
    df_T5.to_csv(mapping_file_T5, index=False)
    df_T6.to_csv(mapping_file_T6, index=False)

    print(f"Fold calculation took {time.time() - start:.08} seconds.")
    print(f"Fold claculation done.")


if __name__ == "__main__":
    main()
