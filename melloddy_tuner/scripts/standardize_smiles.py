from melloddy_tuner.utils import hash_reference_set
import os
import json
import argparse
import logging
from pathlib import Path
import time
from typing import Tuple


from pandas import DataFrame


import melloddy_tuner as tuner

from melloddy_tuner.utils.standardizer import Standardizer
from melloddy_tuner.utils.df_transformer import DfTransformer
from melloddy_tuner.utils.config import ConfigDict, SecretDict
from melloddy_tuner.utils.helper import (
    load_config,
    load_key,
    make_dir,
    create_log_files,
    read_csv,
    sanity_check_uniqueness,
    save_run_report,
    validate_T2,
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
        help="path of the standardized structure input file",
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


def prepare_data_transformer(number_cpu: int) -> DfTransformer:
    method_params = ConfigDict.get_parameters()["standardization"]
    st = Standardizer.from_param_dict(method_param_dict=method_params, verbosity=0)
    outcols = ["canonical_smiles", "success", "error_message"]
    out_types = ["object", "bool", "object"]

    return DfTransformer(
        st,
        input_columns={"smiles": "smiles"},
        output_columns=outcols,
        output_types=out_types,
        success_column="success",
        nproc=number_cpu,
        verbosity=0,
    )


def prepare(args):
    """
    Prepare output directories and instantiate df tansformer object for scaffold based folding

    Args:
        args (dict): argparser arguments

    Returns:
        Tuple(Path, DfTransformer): Path to output directory and instatitaed DfTranfomer for sccaffold folding


    """
    output_dir = make_dir(
        args, "results_tmp", "standardization", args["non_interactive"]
    )
    create_log_files(output_dir)
    load_config(args)
    dt = prepare_data_transformer(args["number_cpu"])
    return output_dir, dt


def run(df, dt):
    """General warpper function to claculate folds by scaffold

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
    output_dir, dt = prepare(args)
    dict_report = {}
    passed_l = []
    hash_reference_set.main(args)
    dict_report["run_parameters"] = args

    print("Start standardizing structures.")

    input_file = args["structure_file"]
    print("Check uniqueness of T2.")
    dict_structures = {}
    dict_sanity = {}
    df_T2 = read_csv(input_file)
    validate_T2(df_T2)
    dict_structures["input_smi"] = df_T2.shape[0]
    passed, dict_unique = sanity_check_uniqueness(
        df_T2, colname="input_compound_id", filename="T2"
    )
    passed_l.append(passed)
    dict_sanity["uniqueness"] = dict_unique
    print(f"Sanity checks took {time.time() - start:.08} seconds.")
    dict_report["sanity_checks"] = dict_sanity
    if False in passed_l:
        save_run_report(args, dict_report, mode="standardize_smiles")
        exit("Found error. Please check the report.")
    else:
        print(f"Sanity checks passed.")

    print("Start standardizing structures.")
    df_smi, df_smi_failed = run(df_T2, dt)
    output_file = os.path.join(output_dir, "T2_standardized.csv")
    error_file = os.path.join(output_dir, "T2_standardized.FAILED.csv")

    dict_structures["standardized_smi"] = df_smi.shape[0]
    dict_structures["failed_smi"] = df_smi_failed.shape[0]
    dict_report["smiles_standardization"] = dict_structures
    df_smi.to_csv(output_file, index=False)
    df_smi_failed.to_csv(error_file, index=False)
    save_run_report(args, dict_report, mode="standardize_smiles")

    print(f"Standardization took {time.time() - start:.08} seconds.")
    print(f"Standardization done.")


if __name__ == "__main__":
    main()
