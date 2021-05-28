import os
from pathlib import Path
from melloddy_tuner import utils
from melloddy_tuner.utils import helper, version
from melloddy_tuner.utils.config import ConfigDict
from melloddy_tuner.utils.standardizer import Standardizer
import time
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import multiprocessing


import melloddy_tuner
from melloddy_tuner.scripts import (
    activity_data_formatting,
    aggregate_values,
    calculate_descriptors,
    calculate_lsh_folds,
    calculate_scaffold_folds,
    filter_classification,
    filter_regression,
    standardize_smiles,
    apply_thresholding,
)
from melloddy_tuner.scripts import csv_2_mtx

from melloddy_tuner.utils import chem_utils, hash_reference_set
from melloddy_tuner.utils.helper import (
    load_config,
    load_key,
    make_dir,
    map_2_cont_id,
    read_csv,
    sanity_check_assay_sizes,
    sanity_check_assay_type,
    sanity_check_compound_sizes,
    sanity_check_uniqueness,
    save_df_as_csv,
    read_input_file,
    save_mtx_as_npy,
)
from sys import platform

if platform == "darwin":
    multiprocessing.set_start_method("fork", force=True)


#######################################
"""
Commandline Interface to run MELLODDY-TUNER:
The tool can execute the following commands:
(1) "standard_smiles" : Standardization of given input filer (T2).
(2) "assign_fold": Assign folds by scaffold network.
(3) "calc_desc": Calcuelate molecular descriptors.
(4) "agg_activity_data": Aggregate input activity data (T0, T1).
(5) "apply_thresholding": Apply thresholding to classification data.
(6) "filter_classification_data": Filter classification data.
(7) "filter_regression_data": Filter regression data.
(8) "make_matrices": Create sparse matrices from processed dataframes, ready for SparseChem.
(9) "prepare_4_training": Entire pipeline  for training processing including function (1) - (8). 
(10) "prepare_4_prediction": Entire pipeline for prediction including function 1,3 and 8.
"""


parser = ArgumentParser(
    description=f"MELLODDY-TUNER: Standardardization Tool for IMI MELLODDY (Version: {version.__version__})"
)
subparsers = parser.add_subparsers(
    title="subcommands",
    help="Use 'tunercli <subcommand> --help' for details about the given subcommand",
)


#######################################

"""
Standardize SMILES Subparser
"""
standardize = subparsers.add_parser(
    "standardize_smiles", description="Standardize SMILES structures"
)
standardize.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the structure input file",
    required=True,
)
standardize.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=True
)
standardize.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=True
)
standardize.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=True,
)
standardize.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=True
)
standardize.add_argument(
    "-n",
    "--number_cpu",
    type=int,
    help="number of CPUs for calculation (default: 1)",
    default=1,
)
standardize.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)
standardize.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
    default=False,
)


def do_standardize_smiles(args):
    """Standardize smiles

    Args:
        args (Namespace): subparser arguments
    """
    # hash_reference_set.main
    standardize_smiles.main(vars(args))


standardize.set_defaults(func=do_standardize_smiles)


#######################################
assign_fold = subparsers.add_parser("assign_fold", description="fold assignment")

assign_fold.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the standardized structure input file",
    required=True,
)
assign_fold.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=True
)
assign_fold.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=True
)
assign_fold.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=True,
)
assign_fold.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=True
)
assign_fold.add_argument(
    "-n",
    "--number_cpu",
    type=int,
    help="number of CPUs for calculation (default: 1 CPUs)",
    default=1,
)
assign_fold.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)
assign_fold.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
    default=False,
)


def do_fold_assignment(args):
    """Standardize smiles

    Args:
        args (Namespace): subparser arguments
    """
    # hash_reference_set.main
    calculate_scaffold_folds.main(vars(args))


assign_fold.set_defaults(func=do_fold_assignment)
#######################################
"""
 Calculate Descriptor Subparser
"""

calc_desc = subparsers.add_parser(
    "calculate_descriptors", description="Calculate descriptors"
)

calc_desc.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the structure input file containing standardized smiles and optional fold ID",
    required=True,
)
calc_desc.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=True
)
calc_desc.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=True
)
calc_desc.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=True,
)
calc_desc.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=True
)
calc_desc.add_argument(
    "-n",
    "--number_cpu",
    type=int,
    help="number of CPUs for calculation (default: 1)",
    default=1,
)
calc_desc.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)
calc_desc.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
    default=False,
)


def do_calculate_desc(args):
    """Calculate descriptors and assign folds.

    Args:
        args (Namespace): subparser arguments
    """
    calculate_descriptors.main(vars(args))


calc_desc.set_defaults(func=do_calculate_desc)
#######################################
# Descriptor calculation and Locality sensitive hashing based fold assignment

desc_lsh = subparsers.add_parser(
    "assign_lsh_fold", description="Run descriptor calculation and LSH based folding."
)
desc_lsh.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the structure input file",
    required=True,
)
desc_lsh.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=True
)
desc_lsh.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=True
)
desc_lsh.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=True,
)
desc_lsh.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=True
)
desc_lsh.add_argument(
    "-n",
    "--number_cpu",
    type=int,
    help="number of CPUs for calculation (default: 2 CPUs)",
    default=2,
)
desc_lsh.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)
desc_lsh.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
    default=False,
)


def do_calculate_desc_lsh(args):
    """Calculate descriptors and assign folds based on locality sensitive hashing.

    Args:
        args (Namespace): subparser arguments
    """
    calculate_lsh_folds.main(vars(args))


desc_lsh.set_defaults(func=do_calculate_desc_lsh)

#######################################
"""
 Aggregate activty data
"""

agg_act_data = subparsers.add_parser(
    "agg_activity_data", description="Aggregation of activity data"
)

agg_act_data.add_argument(
    "-assay",
    "--assay_file",
    type=str,
    help="path of the assay metadata file T0",
    required=True,
)
agg_act_data.add_argument(
    "-a",
    "--activity_file",
    type=str,
    help="path of the activity data file T1",
    required=True,
)
agg_act_data.add_argument(
    "-mt",
    "--mapping_table",
    type=str,
    help="path of the mapping table T5",
    required=True,
)
agg_act_data.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=True
)
agg_act_data.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=True
)
agg_act_data.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=True,
)
agg_act_data.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=True
)
agg_act_data.add_argument(
    "-n",
    "--number_cpu",
    type=int,
    help="number of CPUs for calculation (default: 1)",
    default=1,
)
agg_act_data.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)
agg_act_data.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
    default=False,
)


def do_agg_activity_data(args):
    """Aggregate activity data

    Args:
        args (Namespace): subparser arguments
    """
    aggregate_values.main(vars(args))


agg_act_data.set_defaults(func=do_agg_activity_data)

#######################################
"""
Apply Thresholding
"""

apply_threshold = subparsers.add_parser(
    "apply_thresholding", description="Thresholding of activity data"
)

apply_threshold.add_argument(
    "-assay",
    "--assay_file",
    type=str,
    help="path of the assay metadata file T0",
    required=True,
)
apply_threshold.add_argument(
    "-a",
    "--activity_file",
    type=str,
    help="path of the activity data file T4r",
    required=True,
)
apply_threshold.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=True
)
apply_threshold.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=True
)
apply_threshold.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=True,
)
apply_threshold.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=True
)
apply_threshold.add_argument(
    "-n",
    "--number_cpu",
    type=int,
    help="number of CPUs for calculation (default: 1)",
    default=1,
)
apply_threshold.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)
apply_threshold.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
    default=False,
)


def do_thresholding(args):
    """Apply thresholding

    Args:
        args (Namespace): subparser arguments
    """
    apply_thresholding.main(vars(args))


apply_threshold.set_defaults(func=do_thresholding)

#######################################
"""
Filter classification data
"""

filter_clf = subparsers.add_parser(
    "filter_classification_data", description="filter classification activity data"
)

filter_clf.add_argument(
    "-ca",
    "--classification_activity_file",
    type=str,
    help="path of the classification task data T4c",
    required=True,
)
filter_clf.add_argument(
    "-cw",
    "--classification_weight_table",
    type=str,
    help="path of the classification task definition and metadata T3c",
    required=True,
)
filter_clf.add_argument(
    "-mt",
    "--mapping_table_T5",
    type=str,
    help="path to mapping table T5",
    required=False,
)
filter_clf.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=True
)
filter_clf.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=True
)
filter_clf.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=True,
)
filter_clf.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=True
)
filter_clf.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)
filter_clf.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
    default=False,
)


def do_filtering_clf(args):
    """filter classification data

    Args:
        args (Namespace): subparser arguments
    """
    filter_classification.main(vars(args))


filter_clf.set_defaults(func=do_filtering_clf)

#######################################
"""
Filter regression data
"""

filter_reg = subparsers.add_parser(
    "filter_regression_data", description="filter regression activity data"
)

filter_reg.add_argument(
    "-ra",
    "--regression_activity_file",
    type=str,
    help="path of the (censored) regression task data T4r",
    required=True,
)
filter_reg.add_argument(
    "-rw",
    "--regression_weight_table",
    type=str,
    help="path of the (censored) regression task definition and metadata T3r",
    required=True,
)
filter_reg.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=True
)
filter_reg.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=True
)
filter_reg.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=True,
)
filter_reg.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=True
)
filter_reg.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)
filter_reg.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
    default=False,
)


def do_filtering_reg(args):
    """filter regression data

    Args:
        args (Namespace): subparser arguments
    """
    filter_regression.main(vars(args))


filter_reg.set_defaults(func=do_filtering_reg)


#######################################
"""
Create Sparse Matrices for SparseChem Subparser
"""

sparse_matrices = subparsers.add_parser(
    "make_matrices", description="Formatting of activity data"
)
sparse_matrices.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the processed structure input file T6",
    required=True,
)
sparse_matrices.add_argument(
    "-ac",
    "--activity_file_clf",
    type=str,
    help="path of the processed classification activity file T10c",
)
sparse_matrices.add_argument(
    "-wc",
    "--weight_table_clf",
    type=str,
    help="path of the processed classification weight table file T8c",
)
sparse_matrices.add_argument(
    "-ar",
    "--activity_file_reg",
    type=str,
    help="path of the processed regression activity file T10r",
)
sparse_matrices.add_argument(
    "-wr",
    "--weight_table_reg",
    type=str,
    help="path of the processed regression weight table file T8r",
)
sparse_matrices.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=True
)
sparse_matrices.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=True
)

sparse_matrices.add_argument(
    "-o", "--output_dir", type=str, help="path to output directory", required=True
)
sparse_matrices.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=True
)
sparse_matrices.add_argument(
    "-t",
    "--tag",
    type=str,
    help="tag to identify classifcation with or without auxiliary data",
    required=True,
)
sparse_matrices.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)

sparse_matrices.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
    default=False,
)


def do_make_sparse_matrices(args):
    """Create matrices form dataframes, ready for SparseChem.

    Args:
        args (Namespace): subparser arguments
    """
    csv_2_mtx.main(vars(args))


sparse_matrices.set_defaults(func=do_make_sparse_matrices)


#######################################
"""
Prepare_4_training Pipeline Subparser
"""

prepare = subparsers.add_parser(
    "prepare_4_training",
    description="Standardize structures, calculate descriptors and folds, format activity data, and generate matrices",
)
prepare.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the structure input file",
    required=True,
)
prepare.add_argument(
    "-a", "--activity_file", type=str, help="path of the activity input file"
)
prepare.add_argument(
    "-w", "--weight_table", type=str, help="path of the weight table file"
)
prepare.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=True
)
prepare.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=True
)
prepare.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=True,
)
prepare.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=True
)
prepare.add_argument(
    "-t",
    "--tag",
    choices=["cls", "clsaux"],
    help="tag to identify classifcation with or without auxiliary data. Valid choices: cls or clsaux",
    required=True,
)
prepare.add_argument(
    "-n",
    "--number_cpu",
    type=int,
    help="number of CPUs for calculation (default: 1)",
    default=1,
)
prepare.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)
prepare.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
)
prepare.add_argument(
    "-fm",
    "--folding_method",
    choices=["scaffold", "lsh"],
    help="select fold assignment method, only scaffold or lsh possible.",
    required=True,
)


def do_prepare_training(args):
    """Wrapper to run the entire pipeline for training.

    Args:
        args (Namespace): Subparser argmuents
    #"""
    start_total = time.time()

    start = time.time()
    _args = vars(args)
    if _args["non_interactive"] is True:
        overwriting = True
    else:
        overwriting = False

    num_cpu = _args["number_cpu"]
    # # load parameters and key
    load_config(_args)
    load_key(_args)
    bit_size = melloddy_tuner.utils.config.parameters.get_parameters()["fingerprint"][
        "fold_size"
    ]
    #########
    # Consistency check
    print("Consistency checks of config and key files.")
    hash_reference_set.main(_args)
    #########
    start = time.time()
    tag = _args["tag"]

    print("Reading input data.")
    df_T0 = read_input_file(_args["weight_table"])
    df_T1 = read_input_file(_args["activity_file"])
    df_T2 = read_input_file(_args["structure_file"])
    print("Data loaded.")
    print("Start sanity checks of input data.")
    print("Check assay types in T0.")
    sanity_check_assay_type(df_T0)

    print("Check consistency of input_assay_id between T0 and T1.")
    sanity_check_assay_sizes(df_T0, df_T1)

    print("Check consistency of input_compound_id between T1 and T2.")
    sanity_check_compound_sizes(df_T1, df_T2)

    print("Check uniqueness of T0 and T2.")
    sanity_check_uniqueness(df_T0, colname="input_assay_id", filename="T0")
    sanity_check_uniqueness(df_T2, colname="input_compound_id", filename="T2")
    print(f"Sanity checks took {time.time() - start:.08} seconds.")
    print(f"Sanity checks passed.")

    start = time.time()
    print("Start standardizing structures.")

    # Make directories, load input files
    results_dir = make_dir(_args, "results", None, overwriting)
    output_dir_std, dt_std = standardize_smiles.prepare(_args)

    df_smi, sd_smi_failed = standardize_smiles.run(df_T2, dt_std)
    save_df_as_csv(output_dir_std, df_smi, "T2_standardized")
    save_df_as_csv(output_dir_std, sd_smi_failed, "T2_standardized.FAILED")
    del sd_smi_failed, df_T2
    print(f"Standardization took {time.time() - start:.08} seconds.")
    print(f"Standardization done.")
    df_T5 = pd.DataFrame()
    df_T6 = pd.DataFrame()
    if _args["folding_method"] == "scaffold":
        print("Using scaffold-based fold assignment.")

        output_dir_desc, dt_desc = calculate_descriptors.prepare(_args, overwriting)

        start = time.time()
        print("Start calculating descriptors.")

        df_desc, df_desc_failed = calculate_descriptors.run(df_smi, dt_desc)

        save_df_as_csv(output_dir_desc, df_desc, "T2_descriptors")
        save_df_as_csv(output_dir_desc, df_desc_failed, "T2_descriptors.FAILED")
        del df_smi, df_desc_failed

        print(f"Fingerprint calculation took {time.time() - start:.08} seconds.")
        print(f"Descriptor calculation done.")

        start = time.time()
        print("Start computing folds.")
        output_dir_fold, mapping_table_dir, dt_fold = calculate_scaffold_folds.prepare(
            _args
        )

        df_fold, df_fold_failed = calculate_scaffold_folds.run(df_desc, dt_fold)
        save_df_as_csv(output_dir_fold, df_fold, "T2_folds")
        save_df_as_csv(output_dir_fold, df_fold_failed, "T2_folds.FAILED")
        del df_fold_failed, df_desc
        df_T5, df_T6, df_duplicates = helper.format_dataframe(df_fold)
        save_df_as_csv(mapping_table_dir, df_T5, "T5")
        save_df_as_csv(mapping_table_dir, df_T6, "T6")
        save_df_as_csv(
            output_dir_desc, df_duplicates, "T2_descriptor_vector_id.DUPLICATES"
        )
        del df_duplicates

        print(f"Fold calculation took {time.time() - start:.08} seconds.")
        print(f"Fold calculation done.")

    elif _args["folding_method"] == "lsh":
        print("Using LSH based fold assignment.")
        output_dir_lsh, mapping_table_dir, dt_lsh = calculate_lsh_folds.prepare(
            _args, overwriting
        )

        output_file = os.path.join(output_dir_lsh, "T2_descriptors_lsh.csv")
        error_file = os.path.join(output_dir_lsh, "T2_descriptors_lsh.FAILED.csv")
        dupl_file = os.path.join(output_dir_lsh, "T2_descriptors_lsh.DUPLICATES.csv")
        mapping_file_T5 = os.path.join(mapping_table_dir, "T5.csv")
        mapping_file_T6 = os.path.join(mapping_table_dir, "T6.csv")

        df_desc_lsh, df_desc_lsh_failed = dt_lsh.process_dataframe(df_smi)
        df_desc_lsh.to_csv(output_file, index=False)
        df_desc_lsh_failed.to_csv(error_file, index=False)
        df_T5, df_T6, df_duplicates = helper.format_dataframe(df_desc_lsh)
        df_duplicates.to_csv(dupl_file, index=False)
        df_T5.to_csv(mapping_file_T5, index=False)
        df_T6.to_csv(mapping_file_T6, index=False)
        del df_duplicates
        end = time.time()
        print(
            f"Fingerprint calculation and LSH folding took {end - start:.08} seconds."
        )
        print(f"Descriptor calculation and LSH folding done.")
    else:
        print("Please use scaffold or lsh as folding method.")
        quit()

    start = time.time()

    print("Start aggregating values.")

    output_dir_agg = aggregate_values.prepare(_args, overwriting)

    (
        df_T4r,
        df_failed_range,
        df_failed_aggr,
        df_failed_std,
        df_dup,
        df_T0_upd,
    ) = aggregate_values.aggregate_replicates(
        df_T0, df_T1, df_T5, ConfigDict.get_parameters()["credibility_range"], num_cpu
    )
    df_T4r = df_T4r[
        [
            "input_assay_id",
            "descriptor_vector_id",
            "fold_id",
            "standard_qualifier",
            "standard_value",
        ]
    ]
    save_df_as_csv(
        output_dir_agg,
        df_T4r,
        "T4r",
        [
            "input_assay_id",
            "descriptor_vector_id",
            "fold_id",
            "standard_qualifier",
            "standard_value",
        ],
    )
    save_df_as_csv(
        output_dir_agg,
        df_failed_range,
        "failed_range_T1",
        ["input_compound_id", "input_assay_id", "standard_qualifier", "standard_value"],
    )
    save_df_as_csv(
        output_dir_agg,
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
        output_dir_agg,
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
        output_dir_agg,
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
    save_df_as_csv(output_dir_agg, df_T0_upd, "T0_upd")
    del df_T5, df_failed_range, df_failed_aggr, df_dup, df_T1
    print(f"Replicate aggregation took {time.time() - start:.08} seconds.")
    print(f"Replicate aggregation done.")

    start = time.time()
    print("Start thresholding.")
    output_dir_thres = apply_thresholding.prepare(_args, overwriting)
    df_T0_upd = df_T0_upd.astype({"input_assay_id": "int"})
    df_T4r = df_T4r.astype({"input_assay_id": "int"})
    df_T4c, df_T3c = apply_thresholding.run(df_T0_upd, df_T4r, num_cpu)

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

    df_T4c = df_T4c[columns_T4c]
    df_T3c = df_T3c[columns_T3c]

    save_df_as_csv(output_dir_thres, df_T4c_failed, "T4c.FAILED")
    save_df_as_csv(output_dir_thres, df_T4c, "T4c")
    save_df_as_csv(output_dir_thres, df_T3c, "T3c")

    print(f"Thresholding took {time.time() - start:.08} seconds.")
    print(f"Thresholding done.")

    print("Start filter classification data.")
    start = time.time()

    output_dir_filter_clf = filter_classification.prepare(_args, overwriting)
    T10c, T8c, T4c_filtered_out, T4c_dedup = filter_classification.filter_clf(
        df_T3c,
        df_T4c,
        ConfigDict.get_parameters()["training_quorum"]["classification"],
        ConfigDict.get_parameters()["evaluation_quorum"]["classification"],
        ConfigDict.get_parameters()["initial_task_weights"],
    )

    filter_classification.write_tmp_output(
        output_dir_filter_clf, T10c, T8c, T4c_filtered_out, T4c_dedup
    )

    del df_T4c, df_T3c, T4c_filtered_out, T4c_dedup

    print(f"Classification filtering took {time.time() - start:.08} seconds.")
    print(f"Classification filtering done.")
    print("Start filter regression data.")
    #####
    start = time.time()
    out_dir_filter_reg = filter_regression.prepare(_args, overwriting)

    T10r, T8r, T4r_filtered_out, T4r_dedup = filter_regression.filter_regression_tasks(
        df_T0_upd,
        df_T4r,
        ConfigDict.get_parameters()["training_quorum"]["regression"],
        ConfigDict.get_parameters()["evaluation_quorum"]["regression"],
        ConfigDict.get_parameters()["initial_task_weights"],
        ConfigDict.get_parameters()["censored_downweighting"],
    )
    filter_regression.write_tmp_output(
        out_dir_filter_reg, T10r, T8r, T4r_filtered_out, T4r_dedup
    )
    del df_T0, df_T4r, T4r_filtered_out, T4r_dedup
    print(f"Filtering regression data took {time.time() - start:.08} seconds.")
    print(f"Filtering regression data done.")

    print("Start creating sparse matrices.")

    start = time.time()
    out_dir_matrices, results_dir = csv_2_mtx.prepare(_args, overwriting)

    df_T6_cont, T10c_cont, T10r_cont = csv_2_mtx.get_cont_id(df_T6, T10c, T10r)
    df_T11 = df_T6_cont[["cont_descriptor_vector_id", "fold_id", "fp_feat"]]

    save_df_as_csv(results_dir, T10c_cont, "T10c_cont")
    save_df_as_csv(results_dir, T10r_cont, "T10r_cont")
    save_df_as_csv(results_dir, df_T6_cont, "T6_cont")

    csv_2_mtx.save_csv_output(out_dir_matrices, tag, T8c, T8r)
    del df_T6, df_T6_cont, T10r, T10c

    (
        x_matrix,
        fold_vector,
        y_matrix_clf,
        y_matrix_reg,
        censored_mask,
    ) = csv_2_mtx.make_matrices(df_T11, T10c_cont, T10r_cont, bit_size)
    del df_T11, T10c_cont, T10r_cont
    y_matrix_clf.data = np.nan_to_num(y_matrix_clf.data, copy=False)
    y_matrix_clf.eliminate_zeros()

    csv_2_mtx.save_npy_matrices(
        out_dir_matrices,
        tag,
        x_matrix,
        fold_vector,
        y_matrix_clf,
        y_matrix_reg,
        censored_mask,
    )

    print(f"Formatting to matrices took {time.time() - start:.08} seconds.")
    end = time.time()
    print(f"Overall processing took {end - start_total:.08} seconds.")
    print(f"Files are ready for SparseChem.")


prepare.set_defaults(func=do_prepare_training)


#######################################
"""
Prepare_4_prediction Pipeline Subparser
"""

prediction = subparsers.add_parser(
    "prepare_4_prediction",
    description="Standardize structures, calculate descriptors and folds, format activity data, and generate matrices",
)
prediction.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the structure input file",
    required=True,
)
prediction.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=True
)
prediction.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=True
)
prediction.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=True,
)
prediction.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=True
)
prediction.add_argument(
    "-n",
    "--number_cpu",
    type=int,
    help="number of CPUs for calculation (default: 1)",
    default=1,
)
prediction.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)
prediction.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
)


def do_prepare_prediction(args):
    """Wrapper to run the entire pipeline for training.

    Args:
        args (Namespace): Subparser argmuents
    """
    start = time.time()
    _args = vars(args)
    if _args["non_interactive"] is True:
        overwriting = True
    else:
        overwriting = False

    num_cpu = _args["number_cpu"]
    # load parameters and key
    load_config(_args)
    load_key(_args)
    bit_size = melloddy_tuner.utils.config.parameters.get_parameters()["fingerprint"][
        "fold_size"
    ]
    #########
    # Consistency check
    print("Consistency checks of config and key files.")
    hash_reference_set.main(_args)
    #########
    print("Prepare for prediction.")

    ######
    df = read_input_file(_args["structure_file"])
    # Make directories, load input files
    output_dir_std, dt_std = standardize_smiles.prepare(_args)

    df_smi, df_smi_failed = standardize_smiles.run(df, dt_std)
    output_dir_desc, dt_desc = calculate_descriptors.prepare(_args, overwriting)
    df_desc, df_desc_failed = calculate_descriptors.run(df_smi, dt_desc)
    df_desc_c = df_desc.copy()
    df_desc_c.loc[:, "descriptor_vector_id"] = (
        df_desc_c.groupby("input_compound_id").ngroup().replace(-1, np.nan).add(1)
    )
    df_T6 = df_desc_c[["descriptor_vector_id", "fp_feat", "fp_val"]]
    out_dir_matrices, results_dir = csv_2_mtx.prepare(_args, overwriting)

    df_T11 = map_2_cont_id(df_T6, "descriptor_vector_id").sort_values(
        "cont_descriptor_vector_id"
    )

    save_df_as_csv(results_dir, df_T11, "T11_pred")
    x_matrix = csv_2_mtx.matrix_from_strucutres(df_T11, bit_size)
    save_mtx_as_npy(x_matrix, out_dir_matrices, "pred_x")
    print(f"Preparation took {time.time() - start:.08} seconds.")
    print(f"Prediction preparation done.")


prediction.set_defaults(func=do_prepare_prediction)


def main():
    args = parser.parse_args()
    if "func" in args:
        args.func(args)
    else:
        parser.print_help()
    pass


if __name__ == "__main__":
    main()
