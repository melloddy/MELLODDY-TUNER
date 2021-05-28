"""
Common code for data preparation for the MELLODDY project.

 Calculation of descriptors (fingerprints) with RDKit

"""

import argparse
from melloddy_tuner.utils.df_transformer import DfTransformer
from melloddy_tuner.utils.lsh_folding import LSHFoldingCalculator
from melloddy_tuner.utils.config import ConfigDict, SecretDict
import os
import time
from pathlib import Path
from typing import Tuple

from melloddy_tuner.utils import hash_reference_set
from melloddy_tuner.utils.chem_utils import (
    output_descriptor_duplicates,
    output_mapping_table,
    output_processed_descriptors,
    run_fingerprint,
)
from melloddy_tuner.utils.folding import LSHFolding
from melloddy_tuner.utils.formatting import ActivityDataFormatting
from melloddy_tuner.utils.helper import (
    concat_desc_folds,
    create_log_files,
    load_config,
    load_key,
    make_dir,
    read_input_file,
    save_df_as_csv,
)
from pandas import DataFrame
import pandas as pd


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

    parser.add_argument(
        "-p",
        "--prediction_only",
        help="Preprocess only chemical structures for prediction mode",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-rs",
        "--reference_set",
        type=str,
        help="path of the reference set file for unit tests",
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
    output_dir_lsh = make_dir(args, "results_tmp", "lsh_folding", overwriting)
    mapping_table_dir = make_dir(args, "mapping_table", None, overwriting)
    create_log_files(output_dir_lsh)
    create_log_files(mapping_table_dir)
    load_config(args)
    load_key(args)
    method_params_fp = ConfigDict.get_parameters()["fingerprint"]
    method_params_lsh = ConfigDict.get_parameters()["lsh"]
    method_params = {**method_params_fp, **method_params_lsh}
    key = SecretDict.get_secrets()["key"]
    lshf = LSHFoldingCalculator.from_param_dict(
        secret=key, method_param_dict=method_params, verbosity=0
    )
    outcols = ["fp_feat", "fp_val", "fold_id", "success", "error_message"]
    out_types = ["object", "object", "object", "bool", "object"]
    dt = DfTransformer(
        lshf,
        input_columns={"canonical_smiles": "smiles"},
        output_columns=outcols,
        output_types=out_types,
        success_column="success",
        nproc=args["number_cpu"],
        verbosity=0,
    )
    return output_dir_lsh, mapping_table_dir, dt


# def calc_desc(df: DataFrame, num_cpu: int = 1):
#     """Run fingerprint calculation in multiprocessing mode.

#     Args:
#         df (DataFrame): input dataframe containing standardized smiles in column "canonical_smiles"
#         num_cpu (int, optional): Number of CPU core to use for multiprocessing standardization. Defaults to 1.

#     Returns:
#         list: List of Calculated ECFP fingerprints (features and values) both as numpy arrays.
#     """
#     ecfp = DataFrame()
#     ecfp = run_fingerprint(df['canonical_smiles'], num_cpu)
#     return ecfp


# def folding(ecfp: list):
#     """Run LSH fold assignment

#     Args:
#         ecfp (list): List of ECFP numpy arrays (features and values)

#     Returns:
#         Tuple(DataFrame, DataFrame): Assigned folds as dataframe and dataframe of high entropy bits
#     """
#     start = time.time()
#     folds = DataFrame()
#     df_high_entropy_bits = DataFrame()
#     lsh = LSHFolding()
#     folds = lsh.run_lsh_calculation(ecfp)
#     df_high_entropy_bits = lsh.calc_highest_entropy_bits(ecfp)
#     print(f'LSH folding took {time.time() - start:.08} seconds.')
#     return folds, df_high_entropy_bits


def get_mapping_tables(df_processed: DataFrame, df_concat: DataFrame):
    """Get mapping mapping tables from processed dataframes

    Args:
        df_processed (DataFrame): processed descriptor dataframe
        df_concat (DataFrame): decriptor and fold containing dataframe

    Returns:
        Tuple(DataFrame, DataFrame, DataFrame): mapping_table_T5, mapping_table_T6, and mapping_table_T10
    """
    mapping_table_T5 = output_mapping_table(
        df_concat, col_to_keep=["input_compound_id", "fold_id", "descriptor_vector_id"]
    )

    mapping_table_T6 = output_mapping_table(
        df_concat, col_to_keep=["descriptor_vector_id", "fp_feat", "fp_val", "fold_id"]
    )

    return mapping_table_T5, mapping_table_T6


# def format_output(df: DataFrame, ecfp: list, folds: DataFrame):
#     """Add ecfp to input dataframe, concatenate with fold id, and identify duplicates

#     Args:
#         df (DataFrame): Input dataframe containing standardized smiles
#         ecfp (list): calculated ecfp features and values
#         folds (DataFrame): assigned fold id dataframe

#     Returns:
#         Tuple(DataFrame, DataFrame, DataFrame): Input dataframe with ecfp information, input dataframe with ecfp and fold information, dataframe with duplicate descriptors
#     """
#     df_processed = output_processed_descriptors(ecfp, df)
#     df_out = pd.concat([df_processed, folds], axis=1)

#     return df_out
# #     df_duplicates = output_descriptor_duplicates(df_processed)
#     # return df_processed, df_concat, df_duplicates


def run(df, dt):
    """General warpper function to claculate fingerprints and

    Args:
        df (DataFrame): Dataframe with standardized smiles

    Returns:
        Tuple (DataFrame, DataFrame): a datframe with successfully calculated fold information, datafarem with failed molecules
    """
    return dt.process_dataframe(df)


def format_dataframe(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Create unique descriptor vector id and generate mapping table T5 and save duplicates.

    Args:
        df (DataFrame): dataframe with descriptor features and values, as well as input compound id.

    Returns:
        Tuple[Dataframe, Dataframe]: T5 mapping table, descriptor-based duplicates
    """
    # identify duplicated fingerprint, create unique descriptor vector ID for them,
    # and sort them according to the new descriptor ID
    df["descriptor_vector_id"] = df.groupby(["fp_feat", "fp_val", "fold_id"]).ngroup()
    df_grouped = df.drop_duplicates(
        ["descriptor_vector_id", "fp_feat", "fp_val", "fold_id"]
    ).sort_values("descriptor_vector_id")
    df_structure_duplicates = pd.DataFrame(
        df,
        columns=[
            "input_compound_id",
            "canonical_smiles",
            "fp_feat",
            "fp_val",
            "fold_id",
            "descriptor_vector_id",
        ],
    )
    df_structure_duplicates = df_structure_duplicates[
        df_structure_duplicates.duplicated(["descriptor_vector_id"], keep=False)
    ]
    df_structure_duplicates = df_structure_duplicates.sort_values(
        "descriptor_vector_id"
    )

    return df_grouped, df_structure_duplicates


# def get_T11(df: DataFrame):
#     """Map processed dataframe to continuous identifiers and generate T11

#     Args:
#         df (DataFrame): Processed dataframe containing descriptor  and fold ids.

#     Returns:
#         DataFrame: DataFrame T11
#     """
#     df_remapped = ActivityDataFormatting.map_2_cont_id(
#         df, 'descriptor_vector_id').sort_values('cont_descriptor_vector_id')
#     return df_remapped


# def write_tmp_output(out_dir: Path, df_duplicates: DataFrame, df_high_entropy_bits: DataFrame):
#     """Wrapper to save csv files to results_tmp

#     Args:
#         out_dir (Path): Path to output subfolder "restults_tmp"
#         df_duplicates (DataFrame): DataFrame containing duplicated entries based on descriptor ids.
#         df_high_entropy_bits (DataFrame): DataFrame with high entropy bits from given input file.
#     """

#     save_df_as_csv(out_dir, df_duplicates,  "desc_duplicates")
#     save_df_as_csv(out_dir, df_high_entropy_bits,  "high_entropy_bits")


# def write_output(out_dir, mapping_table_T5, mapping_table_T6):
#     """Wrapper to save csv files to results

#     Args:
#         out_dir (Path): Path to output subfolder "results"
#         mapping_table_T5 (DataFrame): DataFrame of mapping table T5
#         mapping_table_T6 (DataFrame): DataFrame of mapping table T6
#         mapping_table_T10 (DataFrame): DataFrame of mapping table T10
#     """

#     save_df_as_csv(out_dir, mapping_table_T5,  "mapping_table_T5")
#     save_df_as_csv(out_dir, mapping_table_T6,  "mapping_table_T6")


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
    num_cpu = args["number_cpu"]
    load_config(args)
    load_key(args)
    print("Consistency checks of config and key files.")
    hash_reference_set.main(args)
    print("Start calculating descriptors and assign LSH folds.")
    output_dir_lsh, mapping_table_dir, dt = prepare(args, overwriting)

    input_file = args["structure_file"]
    output_file = os.path.join(output_dir_lsh, "T2_descriptors_lsh.csv")
    error_file = os.path.join(output_dir_lsh, "T2_descriptors_lsh.FAILED.csv")
    dupl_file = os.path.join(output_dir_lsh, "T2_descriptors_lsh.DUPLICATES.csv")
    mapping_file_T5 = os.path.join(mapping_table_dir, "T5.csv")
    mapping_file_T6 = os.path.join(mapping_table_dir, "T6.csv")

    df = pd.read_csv(input_file)
    df_processed, df_failed = dt.process_dataframe(df)
    df_processed.to_csv(output_file, index=False)
    df_failed.to_csv(error_file, index=False)
    df_grouped, df_desc_dupl = format_dataframe(df_processed)
    # col_T5 = ["input_compound_id", "fold_id"]
    # df_T5 = pd.merge(df_processed[col_T5], df_grouped[['input_compound_id', 'descriptor_vector_id', 'fold_id']], on=[
    #                 "input_compound_id", "fold_id"], how="left")
    df_T5 = pd.merge(
        df_processed[["input_compound_id", "fp_feat", "fp_val", "fold_id"]],
        df_grouped[["fp_feat", "fp_val", "descriptor_vector_id", "fold_id"]],
        on=["fp_feat", "fp_val", "fold_id"],
        how="left",
    )[["input_compound_id", "fold_id", "descriptor_vector_id"]].reset_index(drop=True)
    df_T6 = df_grouped[["descriptor_vector_id", "fp_feat", "fp_val", "fold_id"]]
    df_desc_dupl.to_csv(dupl_file, index=False)
    df_T5.to_csv(mapping_file_T5, index=False)
    df_T6.to_csv(mapping_file_T6, index=False)
    end = time.time()
    print(f"Fingerprint calculation and LSH folding took {end - start:.08} seconds.")
    print(f"Descriptor calculation and LSH folding done.")
    # df = read_input_file(args["structure_file"])
    # ecfp = calc_desc(df, num_cpu)

    # folds, df_high_entropy_bits = folding(ecfp)
    # df_high_entropy_bits.to_csv(entropy_file, index=False)
    # # df_processed, df_concat, df_duplicates = format_output(df, ecfp, folds)
    # df_processed = format_output(df, ecfp, folds)
    # df_processed.to_csv(output_file, index=False)

    # df_grouped, df_desc_dupl = format_dataframe(df_processed)
    # col_T5 = ["input_compound_id", "fold_id"]
    # df_T5 = pd.merge(df_processed[col_T5], df_grouped[['input_compound_id', 'descriptor_vector_id', 'fold_id']], on=["input_compound_id", "fold_id"], how="left")
    # df_T6 = df_grouped[['descriptor_vector_id', 'fp_feat', 'fp_val', 'fold_id']]
    # print(df_T5)

    # df_desc_dupl.to_csv(dupl_file, index=False)
    # df_T5.to_csv(mapping_file_T5, index=False)
    # df_T6.to_csv(mapping_file_T6, index=False)
    # # mapping_table_T5, mapping_table_T6= get_mapping_tables(
    # #     df_processed, df_concat)
    # # write_tmp_output(output_dir, df_duplicates, df_high_entropy_bits)
    # # write_output(mapping_table_dir, mapping_table_T5,
    # #              mapping_table_T6)
    # end = time.time()
    # print(
    #     f'Fingerprint calculation and LSH clustering took {end - start:.08} seconds.')
    # print(f'Descriptor calculation done.')


if __name__ == "__main__":
    main()
