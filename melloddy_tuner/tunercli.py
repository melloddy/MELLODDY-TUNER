import os, sys
from pathlib import Path
from typing import Dict, Tuple

from pandas.core.frame import DataFrame
from melloddy_tuner.utils import version
from melloddy_tuner.utils.config import ConfigDict
import time
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import multiprocessing

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

from melloddy_tuner.utils import hash_reference_set, config, helper, folder_s3_ready
from melloddy_tuner.utils.helper import (
    load_config,
    load_key,
    make_dir,
    map_2_cont_id,
    read_csv,
    sanity_check_input_assay_id,
    sanity_check_assay_sizes,
    sanity_check_assay_type,
    sanity_check_compound_sizes,
    sanity_check_uniqueness,
    sanity_check_binary,
    save_df_as_csv,
    read_input_file,
    allign_compound_sizes,
    read_input_files,
    save_mtx_as_npy,
    save_run_args,
    read_run_params,
    save_run_report,
    concatenate_T_files,
    counts_per_type,
    validate_T0,
    validate_T1,
    validate_T2
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
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")
bool_stand_smi = '--run_parameters' not in sys.argv and '-p' not in sys.argv
standardize.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the structure input file",
    required=bool_stand_smi,
)
standardize.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=bool_stand_smi
)
standardize.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_stand_smi
)
standardize.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=bool_stand_smi,
)
standardize.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_stand_smi
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
    tmp_args = vars(args)
    
    
    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="standardize_smiles")
    
    standardize_smiles.main(_args)


standardize.set_defaults(func=do_standardize_smiles)

#######################################
"""
 Calculate Descriptor Subparser
"""

calc_desc = subparsers.add_parser(
    "calculate_descriptors", description="Calculate descriptors"
)

calc_desc.add_argument(
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")
bool_stand_desc = '--run_parameters' not in sys.argv and '-p' not in sys.argv

calc_desc.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the structure input file containing standardized smiles and optional fold ID",
    required=bool_stand_desc,
)
calc_desc.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=bool_stand_desc
)
calc_desc.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_stand_desc
)
calc_desc.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=bool_stand_desc,
)
calc_desc.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_stand_desc
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
    tmp_args = vars(args)
    
    
    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="calculate_descriptors")
    
    calculate_descriptors.main(_args)


calc_desc.set_defaults(func=do_calculate_desc)

#######################################
assign_fold = subparsers.add_parser("assign_fold", description="fold assignment")

assign_fold.add_argument(
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")
bool_assign_fold= '--run_parameters' not in sys.argv and '-p' not in sys.argv


assign_fold.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the standardized structure input file",
    required=bool_assign_fold,
)
assign_fold.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=bool_assign_fold
)
assign_fold.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_assign_fold
)
assign_fold.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=bool_assign_fold,
)
assign_fold.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_assign_fold
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
    tmp_args = vars(args)
    
    
    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="assign_folds")

    calculate_scaffold_folds.main(_args)


assign_fold.set_defaults(func=do_fold_assignment)

#######################################
# Descriptor calculation and Locality sensitive hashing based fold assignment

desc_lsh = subparsers.add_parser(
    "assign_lsh_fold", description="Run descriptor calculation and LSH based folding."
)

desc_lsh.add_argument(
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")

bool_desc_lsh =  '--run_parameters' not in sys.argv and '-p' not in sys.argv
desc_lsh.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the structure input file",
    required=bool_desc_lsh,
)
desc_lsh.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=bool_desc_lsh
)
desc_lsh.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_desc_lsh
)
desc_lsh.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=bool_desc_lsh,
)
desc_lsh.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_desc_lsh
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
    tmp_args = vars(args)
    
    
    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="assign_lsh_folds")
    calculate_lsh_folds.main(_args)


desc_lsh.set_defaults(func=do_calculate_desc_lsh)

#######################################
"""
 Aggregate activty data
"""

agg_act_data = subparsers.add_parser(
    "agg_activity_data", description="Aggregation of activity data"
)

agg_act_data.add_argument(
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")

bool_aggr =  '--run_parameters' not in sys.argv and '-p' not in sys.argv

agg_act_data.add_argument(
    "-assay",
    "--assay_file",
    type=str,
    help="path of the assay metadata file T0",
    required=bool_aggr,
)
agg_act_data.add_argument(
    "-a",
    "--activity_file",
    type=str,
    help="path of the activity data file T1",
    required=bool_aggr,
)
agg_act_data.add_argument(
    "-mt",
    "--mapping_table",
    type=str,
    help="path of the mapping table T5",
    required=bool_aggr,
)
agg_act_data.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=bool_aggr
)
agg_act_data.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_aggr
)
agg_act_data.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=bool_aggr,
)
agg_act_data.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_aggr
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
    tmp_args = vars(args)
    
    
    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="aggregate_values")

    aggregate_values.main(_args)


agg_act_data.set_defaults(func=do_agg_activity_data)

#######################################
"""
Apply Thresholding
"""

apply_threshold = subparsers.add_parser(
    "apply_thresholding", description="Thresholding of activity data"
)


apply_threshold.add_argument(
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")

bool_thres =  '--run_parameters' not in sys.argv and '-p' not in sys.argv


apply_threshold.add_argument(
    "-assay",
    "--assay_file",
    type=str,
    help="path of the assay metadata file T0",
    required=bool_thres,
)
apply_threshold.add_argument(
    "-a",
    "--activity_file",
    type=str,
    help="path of the activity data file T4r",
    required=bool_thres,
)
apply_threshold.add_argument(
        "-ct",
        "--catalog_file",
        type=str,
        help="path of the reference catalog  file T_cat",
        required=bool_thres
    )
apply_threshold.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=bool_thres
)
apply_threshold.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_thres
)
apply_threshold.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=bool_thres,
)
apply_threshold.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_thres
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
    tmp_args = vars(args)


    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="apply_thresholding")
    apply_thresholding.main(_args)


apply_threshold.set_defaults(func=do_thresholding)

#######################################
"""
Filter classification data
"""

filter_clf = subparsers.add_parser(
    "filter_classification_data", description="filter classification activity data"
)

filter_clf.add_argument(
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")

bool_clf=  '--run_parameters' not in sys.argv and '-p' not in sys.argv


filter_clf.add_argument(
    "-ca",
    "--classification_activity_file",
    type=str,
    help="path of the classification task data T4c",
    required=bool_clf,
)
filter_clf.add_argument(
    "-cw",
    "--classification_weight_table",
    type=str,
    help="path of the classification task definition and metadata T3c",
    required=bool_clf,
)
filter_clf.add_argument(
    "-mt",
    "--mapping_table_T5",
    type=str,
    help="path to mapping table T5",
    required=bool_clf,
)

filter_clf.add_argument(
        "-ct",
        "--catalog_file",
        type=str,
        help="path of the reference catalog  file T_cat",
        required=bool_clf
)

filter_clf.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=bool_clf
)

filter_clf.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_clf
)
filter_clf.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=bool_clf,
)
filter_clf.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_clf
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
    tmp_args = vars(args)


    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="filter_classification")

    filter_classification.main(_args)


filter_clf.set_defaults(func=do_filtering_clf)

#######################################
"""
Filter regression data
"""

filter_reg = subparsers.add_parser(
    "filter_regression_data", description="filter regression activity data"
)

filter_reg.add_argument(
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")

bool_reg=  '--run_parameters' not in sys.argv and '-p' not in sys.argv

filter_reg.add_argument(
    "-ra",
    "--regression_activity_file",
    type=str,
    help="path of the (censored) regression task data T4r",
    required=bool_reg,
)
filter_reg.add_argument(
    "-rw",
    "--regression_weight_table",
    type=str,
    help="path of the (censored) regression task definition and metadata T3r",
    required=bool_reg,
)
filter_reg.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=bool_reg
)
filter_reg.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_reg
)
filter_reg.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=bool_reg,
)
filter_reg.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_reg
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
    tmp_args = vars(args)


    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="filter_regression")
    filter_regression.main(_args)


filter_reg.set_defaults(func=do_filtering_reg)


#######################################
"""
Create Sparse Matrices for SparseChem Subparser
"""

sparse_matrices = subparsers.add_parser(
    "make_matrices", description="Formatting of activity data"
)
sparse_matrices.add_argument(
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")

bool_sparse=  '--run_parameters' not in sys.argv and '-p' not in sys.argv

sparse_matrices.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the processed structure input file T6",
    required=bool_sparse,
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
    "-c", "--config_file", type=str, help="path of the config file", required=bool_sparse
)
sparse_matrices.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_sparse
)

sparse_matrices.add_argument(
    "-o", "--output_dir", type=str, help="path to output directory", required=bool_sparse
)
sparse_matrices.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_sparse
)
sparse_matrices.add_argument(
        "-aux",
        "--using_auxiliary",
        choices=["no", "yes"],
        help="tag to identify if auxiliary data is used. Available tags: no or yes",
        required=bool_sparse,
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
    tmp_args = vars(args)


    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="sparse_matrices")

    csv_2_mtx.main(_args)


sparse_matrices.set_defaults(func=do_make_sparse_matrices)

#######################################

make_folders_s3 = subparsers.add_parser(
    "make_folders_s3",
    description="Copy relavent files into a S3-ready folder structure",
)    
make_folders_s3.add_argument(
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")

bool_run_params =  '--run_parameters' not in sys.argv and '-p' not in sys.argv

make_folders_s3.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=bool_run_params
)
make_folders_s3.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_run_params
)

make_folders_s3.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=bool_run_params,
)
make_folders_s3.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_run_params
)
make_folders_s3.add_argument(
        "-aux",
        "--using_auxiliary",
        choices=["no", "yes"],
        help="tag to identify if auxiliary data is used. Available tags: no or yes",
        required=bool_run_params,
)

make_folders_s3.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
)

def do_make_folder_s3(args):
    """Copy relavent files to S3 like fodler structure

    Args:
        args (Namespace): subparser arguments
    """
    tmp_args = vars(args)


    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="sparse_matrices")

    folder_s3_ready.main(_args)


make_folders_s3.set_defaults(func=do_make_folder_s3)

#######################################
"""
Prepare_4_training Pipeline Subparser
"""

prepare = subparsers.add_parser(
    "prepare_4_training",
    description="Standardize structures, calculate descriptors and folds, format activity data, and generate matrices",
)
prepare.add_argument(
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")

bool_run_params =  '--run_parameters' not in sys.argv and '-p' not in sys.argv
prepare.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the structure input file",
    required=bool_run_params,
)
prepare.add_argument(
    "-a", "--activity_files", type=str, help="path of the activity input files", nargs='+'
)
prepare.add_argument(
    "-w", "--weight_tables", type=str, help="path of the weight table files", nargs='+'
)
prepare.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=bool_run_params
)
prepare.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_run_params
)
prepare.add_argument(
        "-ct",
        "--catalog_file",
        type=str,
        help="path of the reference catalog  file T_cat",
        required=bool_run_params
    )

prepare.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=bool_run_params,
)
prepare.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_run_params
)
prepare.add_argument(
        "-aux",
        "--using_auxiliary",
        choices=["no", "yes"],
        help="tag to identify if auxiliary data is used. Available tags: no or yes",
        required=bool_run_params,
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
    required=bool_run_params,
)


def do_prepare_training(args):
    """Wrapper to run the entire pipeline for training.

    Args:
        args (Namespace): Subparser argmuents
    #"""
    start_total = time.time()

    start = time.time()
    dict_report = {}
    tmp_args = vars(args)
    
    
    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="prepare_4_training")
        
    dict_report["run_parameters"] = _args
    

    if _args["non_interactive"] is True:
        overwriting = True
    else:
        overwriting = False

    num_cpu = _args["number_cpu"]
    # # load parameters and key
    load_config(_args)
    load_key(_args)
    bit_size = config.parameters.get_parameters()["fingerprint"][
        "fold_size"
    ]
    #########
    # Consistency check
    print("Consistency checks of config and key files.")
    hash_reference_set.main(_args)
    #########
    start = time.time()
    tag = _args["using_auxiliary"]

    print("Reading input data.")
    if len(_args["weight_tables"]) != len(_args["activity_files"]):
        quit("Please provide the same number of activity files as weight files")
    dict_volume = {}
    df_T0_list = read_input_files(_args["weight_tables"])
    df_T1_list = read_input_files(_args["activity_files"])
    df_T2 = read_input_file(_args["structure_file"])
    validate_T2(df_T2)
    dict_volume["Structures"] = {"Input-Structures": df_T2.shape[0]} 
    print("Data loaded.")
    print("Start sanity checks of input data.")
    dict_sanity = {}
    passed_l = []
    print("Check clashes in T0 files.")
    passed, dict_input_assay = sanity_check_input_assay_id(df_T0_list)
    passed_l.append(passed)
    dict_sanity["input_assay"]= dict_input_assay
    print('Grouping T0 and T1 files.')
    df_T0 = concatenate_T_files(df_T0_list)
    df_T1 = concatenate_T_files(df_T1_list)
    print("Validate T0.")
    validate_T0(df_T0)
    print("Validate T1.")
    validate_T1(df_T1)
    passed, dict_assay_type = sanity_check_assay_type(df_T0, tag)
    passed_l.append(passed)
    dict_sanity["assay-types"]= dict_assay_type
    
    print("Check consistency of input_assay_id between T0 and T1.")
    passed, dict_assay_sizes = sanity_check_assay_sizes(df_T0, df_T1)
    passed_l.append(passed)
    dict_sanity["assay-sizes"]= dict_assay_sizes
    
    print("Check consistency of input_compound_id between T1 and T2.")
    passed, dict_compound_sizes = sanity_check_compound_sizes(df_T1, df_T2)
    passed_l.append(passed)
    dict_sanity["compound_sizes"] = dict_compound_sizes
    df_T2_filtered, df_T2_filtered_failed = allign_compound_sizes(df_T1, df_T2)

    

    print("Check uniqueness of T0 and T2.")
    passed, dict_unique = sanity_check_uniqueness(df_T0, colname="input_assay_id", filename="T0")
    passed_l.append(passed)
    dict_sanity["uniqueness"]= dict_unique
    passed, dict_unique = sanity_check_uniqueness(df_T2_filtered, colname="input_compound_id", filename="T2")
    dict_sanity["uniqueness"] = dict_unique
    passed_l.append(passed)
    print(f"Sanity checks took {time.time() - start:.08} seconds.")
    
    dict_report["sanity_checks"]= dict_sanity
    if False in passed_l:
        save_run_report(_args, dict_report, mode="prepare_4_training")
        exit("Found error. Please check the report.")
    else:
        print(f"Sanity checks passed.")
    
    start = time.time()
    print("Start standardizing structures.")

    # Make directories, load input files
    results_dir = make_dir(_args, "results", None, overwriting)
    output_dir_std, dt_std = standardize_smiles.prepare(_args)
    save_df_as_csv(output_dir_std, df_T2_filtered_failed, "T2_not_T1")
    df_smi, df_smi_failed = standardize_smiles.run(df_T2_filtered, dt_std)
    dict_structures = {}
    dict_structures["standardized_smi"] = df_smi.shape[0]
    dict_structures["failed_smi"] = df_smi_failed.shape[0]
    dict_report["smiles_standardization"] = dict_structures
    save_df_as_csv(output_dir_std, df_smi, "T2_standardized")
    save_df_as_csv(output_dir_std, df_smi_failed, "T2_standardized.FAILED")
    del df_smi_failed, df_T2
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
        dict_desc = {}
        dict_desc["calc_desc"] = df_desc.shape[0]
        dict_desc["failed_desc"] = df_desc_failed.shape[0]
        dict_report["descriptor_calculation"] = dict_desc
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
        dict_fold= {}
        dict_fold["folds"] = df_fold.shape[0]
        dict_fold["failed_folds"] = df_fold_failed.shape[0]
        dict_report["scaffold_folds"] = dict_fold
        save_df_as_csv(output_dir_fold, df_fold, "T2_folds")
        save_df_as_csv(output_dir_fold, df_fold_failed, "T2_folds.FAILED")
        del df_fold_failed, df_desc
        df_T5, df_T6, df_duplicates = helper.format_dataframe(df_fold)
        save_df_as_csv(mapping_table_dir, df_T5, "T5")
        save_df_as_csv(mapping_table_dir, df_T6, "T6")
        save_df_as_csv(
            output_dir_desc, df_duplicates, "T2_descriptor_vector_id.DUPLICATES"
        )
        dict_fold["duplicates"] = df_duplicates.shape[0]
        dict_report["scaffold_folds"] = dict_fold
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
        dict_fold= {}
        dict_fold["folds"] = df_desc_lsh.shape[0]
        dict_fold["failed_folds"] = df_desc_lsh_failed.shape[0]

        df_desc_lsh.to_csv(output_file, index=False)
        df_desc_lsh_failed.to_csv(error_file, index=False)
        df_T5, df_T6, df_duplicates = helper.format_dataframe(df_desc_lsh)
        df_duplicates.to_csv(dupl_file, index=False)
        df_T5.to_csv(mapping_file_T5, index=False)
        df_T6.to_csv(mapping_file_T6, index=False)
        dict_fold["duplicates"] = df_duplicates.shape[0]
        dict_report["scaffold_folds"] = dict_fold
        del df_duplicates
        end = time.time()
        print(
            f"Fingerprint calculation and LSH folding took {end - start:.08} seconds."
        )
        print(f"Descriptor calculation and LSH folding done.")
    else:
        quit("Please use scaffold or lsh as folding method.")

    start = time.time()

    print("Start aggregating values.")
    dict_aggr = {}
    output_dir_agg = aggregate_values.prepare(_args, overwriting)

    (
        df_T4r,
        df_failed_range,
        df_failed_binary,
        df_failed_aggr,
        df_failed_std,
        df_dup,
        df_T0_upd,
    ) = aggregate_values.aggregate_replicates(
        df_T0, df_T1, df_T5, ConfigDict.get_parameters()["credibility_range"], num_cpu
    )
    dict_aggr["passed"] = df_T4r.shape[0]
    dict_aggr["failed_range"] = df_failed_range.shape[0]
    dict_aggr["failed_binary"] = df_failed_binary.shape[0]
    dict_aggr["failed_aggr"] = df_failed_aggr.shape[0]
    dict_aggr["failed_std"] = df_failed_std.shape[0]
    dict_aggr["duplicates"] = df_dup.shape[0]
    dict_report["aggregation"] = dict_aggr
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
        df_failed_binary,
        "failed_binary_T1",
        ["input_compound_id", "input_assay_id",
            "standard_qualifier", "standard_value"],
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
    df_T_cat = read_input_file(_args["catalog_file"])
    dict_thres = {}
    output_dir_thres = apply_thresholding.prepare(_args, overwriting)
    df_T0_upd = df_T0_upd.astype({"input_assay_id": "int"})
    df_T4r = df_T4r.astype({"input_assay_id": "int"})
    df_T4c, df_T3c = apply_thresholding.run(df_T0_upd, df_T4r,df_T_cat,  num_cpu)

    # Write final dataframes (T4c, T3c)
    columns_T3c = [
        "classification_task_id",
        "input_assay_id",
        "assay_type",
        "variance_quorum_OK",
        "use_in_regression",
        "is_binary",
        "is_auxiliary",
        "threshold",
        "threshold_method",
        "direction",
        "catalog_assay_id",
        "catalog_task_id"]
    
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
    dict_thres["passed"] = df_T4c.shape[0]
    dict_thres["failed_labels"] = df_T4c_failed.shape[0]

    dict_report["thresholding"] = dict_thres
    save_df_as_csv(output_dir_thres, df_T4c_failed, "T4c.FAILED")
    save_df_as_csv(output_dir_thres, df_T4c, "T4c")
    save_df_as_csv(output_dir_thres, df_T3c, "T3c")

    print(f"Thresholding took {time.time() - start:.08} seconds.")
    print(f"Thresholding done.")

    print("Start filter classification data.")
    start = time.time()
    dict_clf_filter = {}
    output_dir_filter_clf = filter_classification.prepare(_args, overwriting)
    T10c, T8c, T4c_filtered_out, T4c_dedup = filter_classification.filter_clf(
        df_T3c,
        df_T4c,
        ConfigDict.get_parameters()["training_quorum"]["classification"],
        ConfigDict.get_parameters()["evaluation_quorum"]["classification"],
        ConfigDict.get_parameters()["training_quorum"]["CATALOG-PANEL"],
        ConfigDict.get_parameters()["initial_task_weights"],
    )
    dict_clf_filter["catalog_assays_changed_to_NON-CATALOG-PANEL_assays"] = T8c.loc[(T8c["catalog_check_INFO"] == "missing_ref_task")|(T8c["catalog_check_INFO"] == "failed_quorum"), "input_assay_id"].unique().tolist()
    counts = counts_per_type(T8c, "both",  "clf")
    dict_clf_filter["assays_tasks_per_type"] = counts.set_index("assay_type").to_dict("index")
    dict_clf_filter["dupplicates_desc_task_id"] = T4c_dedup.shape[0]
    dict_clf_filter["filtered_clf_tasks"] = T4c_filtered_out.shape[0]
    T0_cols_wo_exp_thres = [col for col in df_T0.columns if 'expert_threshold' not in col]
    T0_wo_exp_thres = df_T0[T0_cols_wo_exp_thres].copy()
    T0_wo_exp_thres.set_index("input_assay_id", inplace=True)
    T0_T8_diff_col = T0_wo_exp_thres.columns.difference(T8c.columns)
    T0_diff = T0_wo_exp_thres[T0_T8_diff_col].copy().reset_index()
    T8c = T8c.merge(T0_diff, left_on="input_assay_id", right_on="input_assay_id", how="left")
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
    dict_reg_filter = {}
    T10r, T8r, T4r_filtered_out, T4r_dedup = filter_regression.filter_regression_tasks(
        df_T0_upd,
        df_T4r,
        ConfigDict.get_parameters()["training_quorum"]["regression"],
        ConfigDict.get_parameters()["evaluation_quorum"]["regression"],
        ConfigDict.get_parameters()["initial_task_weights"],
        ConfigDict.get_parameters()["censored_downweighting"],
    )
    dict_reg_filter["dupplicates_desc_task_id"] = T4r_dedup.shape[0]
    dict_reg_filter["filtered_reg_data"] = T4r_filtered_out.shape[0]
    
    T8r = filter_regression.filter_regression_evaluation_tasks(
        T8r,
        T10r,
        ConfigDict.get_parameters()["regression_evaluation_task_filter"],)
    filter_regression.write_tmp_output(
        out_dir_filter_reg, T10r, T8r, T4r_filtered_out, T4r_dedup
    )
    del df_T4r, T4r_filtered_out, T4r_dedup
    print(f"Filtering regression data took {time.time() - start:.08} seconds.")
    print(f"Filtering regression data done.")

    print("Start creating sparse matrices.")

    start = time.time()
    out_dir_matrices, results_dir = csv_2_mtx.prepare(_args, overwriting)

    df_T6_cont, T10c_cont, T10r_cont = csv_2_mtx.get_cont_id(df_T6, T10c, T10r)
    df_T11 = df_T6_cont[["cont_descriptor_vector_id", "fold_id", "fp_feat"]]
    dict_clf_filter["cls_datapoints"] = T10c_cont.shape[0]
    dict_clf_filter["cls_tasks"] = T10c_cont["cont_classification_task_id"].nunique()

    dict_reg_filter["reg_datapoints"] = T10r_cont.shape[0]
    dict_reg_filter["censored_reg_datapoints"] = T10r_cont[T10r_cont["is_uncensored"] == False].shape[0]
    dict_reg_filter["reg_tasks"] = T10r_cont["cont_regression_task_id"].nunique()
    
    dict_report["filter_clf"] = dict_clf_filter
    dict_report["filter_reg"] = dict_reg_filter
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
    dict_matrices = {}
    del df_T11, T10c_cont, T10r_cont
    y_matrix_clf.data = np.nan_to_num(y_matrix_clf.data, copy=False)
    y_matrix_clf.eliminate_zeros()
    dict_matrices["x_matrix_feature_dim"] = x_matrix.shape[1]
    dict_matrices["x_matrix_compounds"] =  x_matrix.shape[0]
    dict_matrices["y_matrix_clf_values"] = y_matrix_clf.count_nonzero()
    dict_matrices["y_matrix_reg_values"] = y_matrix_reg.count_nonzero()
    dict_matrices["censored_values"] = censored_mask.count_nonzero()
    dict_matrices["y_matrix_clf_tasks"] = y_matrix_clf.shape[1]
    dict_matrices["y_matrix_reg_tasks"] = y_matrix_reg.shape[1]
    dict_report["matrix_dimensions"] = dict_matrices
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
    run_time = end - start_total
    dict_overall = {}
    dict_overall["run_time_overall"] = run_time
    dict_overall["fraction_censored_reg"] = np.round((dict_matrices["censored_values"]/dict_matrices["y_matrix_reg_values"])*100, 2)
    dict_report["overall_checks"] = dict_overall
    dict_report["run_time_overall"]  = run_time
    print(f"Overall processing took {run_time:.08} seconds.")
    save_run_report(_args, dict_report, mode="prepare_4_training")
    print(f"Files are ready for SparseChem.")
    
    

prepare.set_defaults(func=do_prepare_training)






#######################################
"""
Prepare_structure_data Pipeline Subparser
"""

prepare_structures = subparsers.add_parser(
    "prepare_structure_data",
    description="Standardize structures, calculate descriptors and folds",
)
prepare_structures.add_argument(
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")

bool_run_params =  '--run_parameters' not in sys.argv and '-p' not in sys.argv
prepare_structures.add_argument(
    "-s",
    "--structure_file",
    type=str,
    help="path of the structure input file",
    required=bool_run_params,
)

prepare_structures.add_argument(
    "-a", "--activity_files", type=str, help="path of the activity input files", nargs='+'
)
prepare_structures.add_argument(
    "-w", "--weight_tables", type=str, help="path of the weight table files", nargs='+'
)

prepare_structures.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=bool_run_params
)
prepare_structures.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_run_params
)

prepare_structures.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=bool_run_params,
)
prepare_structures.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_run_params
)
prepare_structures.add_argument(
        "-aux",
        "--using_auxiliary",
        choices=["no", "yes"],
        help="tag to identify if auxiliary data is used. Available tags: no or yes",
        required=bool_run_params,
)
prepare_structures.add_argument(
    "-n",
    "--number_cpu",
    type=int,
    help="number of CPUs for calculation (default: 1)",
    default=1,
)
prepare_structures.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)
prepare_structures.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
)
prepare_structures.add_argument(
    "-fm",
    "--folding_method",
    choices=["scaffold", "lsh"],
    help="select fold assignment method, only scaffold or lsh possible.",
    required=bool_run_params,
)



def do_prepare_structures(args):
    """Wrapper to run the structure processing pipeline.

    Args:
        args (Namespace): Subparser argmuents
    #"""
    start_total = time.time()

    start = time.time()
    dict_report = {}
    tmp_args = vars(args)
    
    
    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="prepare_structures")
        
    dict_report["run_parameters"] = _args
    

    if _args["non_interactive"] is True:
        overwriting = True
    else:
        overwriting = False

    num_cpu = _args["number_cpu"]
    # # load parameters and key
    load_config(_args)
    load_key(_args)
    bit_size = config.parameters.get_parameters()["fingerprint"][
        "fold_size"
    ]
    #########
    # Consistency check
    print("Consistency checks of config and key files.")
    hash_reference_set.main(_args)
    #########
    start = time.time()
    tag = _args["using_auxiliary"]

    print("Reading input data.")
    if len(_args["weight_tables"]) != len(_args["activity_files"]):
        quit("Please provide the same number of activity files as weight files")
    dict_volume = {}
    df_T0_list = read_input_files(_args["weight_tables"])
    df_T1_list = read_input_files(_args["activity_files"])
    df_T2 = read_input_file(_args["structure_file"])
    validate_T2(df_T2)
    dict_volume["Structures"] = {"Input-Structures": df_T2.shape[0]} 
    print("Data loaded.")
    print("Start sanity checks of input data.")
    dict_sanity = {}
    passed_l = []
    print("Check clashes in T0 files.")
    passed, dict_input_assay = sanity_check_input_assay_id(df_T0_list)
    passed_l.append(passed)
    dict_sanity["input_assay"]= dict_input_assay
    print('Grouping T0 and T1 files.')
    df_T0 = concatenate_T_files(df_T0_list)
    df_T1 = concatenate_T_files(df_T1_list)
    print("Check assay types in T0.")
    validate_T0(df_T0)
    validate_T1(df_T1)
    passed, dict_assay_type = sanity_check_assay_type(df_T0, tag)
    passed_l.append(passed)
    dict_sanity["assay-types"]= dict_assay_type
    
    print("Check consistency of input_assay_id between T0 and T1.")
    passed, dict_assay_sizes = sanity_check_assay_sizes(df_T0, df_T1)
    passed_l.append(passed)
    dict_sanity["assay-sizes"]= dict_assay_sizes
    
    print("Check consistency of input_compound_id between T1 and T2.")
    passed, dict_compound_sizes = sanity_check_compound_sizes(df_T1, df_T2)
    passed_l.append(passed)
    dict_sanity["compound_sizes"] = dict_compound_sizes
    df_T2_filtered, df_T2_filtered_failed = allign_compound_sizes(df_T1, df_T2)

    
    

    print("Check uniqueness of T0 and T2.")
    passed, dict_unique = sanity_check_uniqueness(df_T0, colname="input_assay_id", filename="T0")
    passed_l.append(passed)
    dict_sanity["uniqueness"]= dict_unique
    passed, dict_unique = sanity_check_uniqueness(df_T2_filtered, colname="input_compound_id", filename="T2")
    dict_sanity["uniqueness"] = dict_unique
    passed_l.append(passed)
    print(f"Sanity checks took {time.time() - start:.08} seconds.")
    
    dict_report["sanity_checks"]= dict_sanity
    if False in passed_l:
        save_run_report(_args, dict_report, mode="prepare_structure_data")

        exit("Found error. Please check the report.")
    else:
        print(f"Sanity checks passed.")
    
    
    
    
    start = time.time()
    print("Start standardizing structures.")

    # Make directories, load input files
    output_dir_std, dt_std = standardize_smiles.prepare(_args)

    df_smi, df_smi_failed = standardize_smiles.run(df_T2_filtered, dt_std)
    dict_structures = {}
    dict_structures["standardized_smi"] = df_smi.shape[0]
    dict_structures["failed_smi"] = df_smi_failed.shape[0]
    dict_report["smiles_standardization"] = dict_structures
    save_df_as_csv(output_dir_std, df_smi, "T2_standardized")
    save_df_as_csv(output_dir_std, df_smi_failed, "T2_standardized.FAILED")
    del df_smi_failed, df_T2
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
        dict_desc = {}
        dict_desc["calc_desc"] = df_desc.shape[0]
        dict_desc["failed_desc"] = df_desc_failed.shape[0]
        dict_report["descriptor_calculation"] = dict_desc
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
        dict_fold= {}
        dict_fold["folds"] = df_fold.shape[0]
        dict_fold["failed_folds"] = df_fold_failed.shape[0]
        dict_report["scaffold_folds"] = dict_fold
        save_df_as_csv(output_dir_fold, df_fold, "T2_folds")
        save_df_as_csv(output_dir_fold, df_fold_failed, "T2_folds.FAILED")
        del df_fold_failed, df_desc
        df_T5, df_T6, df_duplicates = helper.format_dataframe(df_fold)
        save_df_as_csv(mapping_table_dir, df_T5, "T5")
        save_df_as_csv(mapping_table_dir, df_T6, "T6")
        save_df_as_csv(
            output_dir_desc, df_duplicates, "T2_descriptor_vector_id.DUPLICATES"
        )
        dict_fold["duplicates"] = df_duplicates.shape[0]
        dict_report["scaffold_folds"] = dict_fold
        del df_duplicates

        end = time.time()
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
        dict_fold= {}
        dict_fold["folds"] = df_desc_lsh.shape[0]
        dict_fold["failed_folds"] = df_desc_lsh_failed.shape[0]

        df_desc_lsh.to_csv(output_file, index=False)
        df_desc_lsh_failed.to_csv(error_file, index=False)
        df_T5, df_T6, df_duplicates = helper.format_dataframe(df_desc_lsh)
        df_duplicates.to_csv(dupl_file, index=False)
        df_T5.to_csv(mapping_file_T5, index=False)
        df_T6.to_csv(mapping_file_T6, index=False)
        dict_fold["duplicates"] = df_duplicates.shape[0]
        dict_report["scaffold_folds"] = dict_fold
        del df_duplicates
        end = time.time()
        print(
            f"Fingerprint calculation and LSH folding took {end - start:.08} seconds."
        )
        print(f"Descriptor calculation and LSH folding done.")
    else:
        quit("Please use scaffold or lsh as folding method.")
    
    
    run_time = end - start_total

    dict_report["run_time_overall"]  = run_time
    print(f"Overall structure processing took {run_time:.08} seconds.")
    save_run_report(_args, dict_report, mode="prepare_structure_data")

prepare_structures.set_defaults(func=do_prepare_structures)





#######################################
"""
Prepare_activity_data Pipeline Subparser
"""

prepare_activity_data = subparsers.add_parser(
    "prepare_activity_data",
    description="Format activity data, and generate matrices",
)
prepare_activity_data.add_argument(
    "-p", "--run_parameters", type=str, help="path of the run parameters file (run_params.json)")

bool_run_params =  '--run_parameters' not in sys.argv and '-p' not in sys.argv
prepare_activity_data.add_argument(
        "-mt",
        "--mapping_table",
        type=str,
        help="path of the mapping table T5",
        required=bool_run_params,
    )

prepare_activity_data.add_argument(
        "-T6",
        "--T6_file",
        type=str,
        help="path of the processed structure file T6",
        required=bool_run_params,
    )
prepare_activity_data.add_argument(
    "-a", "--activity_files", type=str, help="path of the activity input files", nargs='+'
)
prepare_activity_data.add_argument(
    "-w", "--weight_tables", type=str, help="path of the weight table files", nargs='+'
)
prepare_activity_data.add_argument(
        "-ct",
        "--catalog_file",
        type=str,
        help="path of the reference catalog  file T_cat",
        required=bool_run_params
    )
prepare_activity_data.add_argument(
    "-c", "--config_file", type=str, help="path of the config file", required=bool_run_params
)
prepare_activity_data.add_argument(
    "-k", "--key_file", type=str, help="path of the key file", required=bool_run_params
)

prepare_activity_data.add_argument(
    "-o",
    "--output_dir",
    type=str,
    help="path to the generated output directory",
    required=bool_run_params,
)
prepare_activity_data.add_argument(
    "-r", "--run_name", type=str, help="name of your current run", required=bool_run_params
)
prepare_activity_data.add_argument(
        "-aux",
        "--using_auxiliary",
        choices=["no", "yes"],
        help="tag to identify if auxiliary data is used. Available tags: no or yes",
        required=bool_run_params,
)
prepare_activity_data.add_argument(
    "-n",
    "--number_cpu",
    type=int,
    help="number of CPUs for calculation (default: 1)",
    default=1,
)
prepare_activity_data.add_argument(
    "-rh",
    "--ref_hash",
    type=str,
    help="path to the reference hash key file provided by the consortium. (ref_hash.json)",
)
prepare_activity_data.add_argument(
    "-ni",
    "--non_interactive",
    help="Enables an non-interactive mode for cluster/server usage",
    action="store_true",
)


def do_prepare_activity_data(args):
    """Wrapper to run the activity formatting  pipeline.

    Args:
        args (Namespace): Subparser argmuents
    """

    start_total = time.time()

    start = time.time()
    dict_report = {}
    tmp_args = vars(args)
    
    
    if os.path.isfile(str(tmp_args["run_parameters"])):
        print("Load run parameters from json file.")
        _args = read_run_params(tmp_args["run_parameters"])
    else:
        _args = save_run_args(tmp_args, mode="prepare_activity_data")
        
    dict_report["run_parameters"] = _args
    

    if _args["non_interactive"] is True:
        overwriting = True
    else:
        overwriting = False

    num_cpu = _args["number_cpu"]
    # # load parameters and key
    load_config(_args)
    load_key(_args)
    bit_size = config.parameters.get_parameters()["fingerprint"][
        "fold_size"
    ]
    #########
    # Consistency check
    print("Consistency checks of config and key files.")
    hash_reference_set.main(_args)
    #########
    start = time.time()
    tag = _args["using_auxiliary"]

    print("Reading input data.")
    if len(_args["weight_tables"]) != len(_args["activity_files"]):
        quit("Please provide the same number of activity files as weight files")
    df_T0_list = read_input_files(_args["weight_tables"])
    
    df_T1_list = read_input_files(_args["activity_files"])
    df_T5 = read_input_file(_args["mapping_table"])
    df_T6 = read_input_file(_args["T6_file"])

    print("Data loaded.")
    print("Start sanity checks of input data.")
    dict_sanity = {}
    passed_l = []
    print("Check clashes in T0 files.")
    passed, dict_input_assay = sanity_check_input_assay_id(df_T0_list)
    passed_l.append(passed)
    dict_sanity["input_assay"]= dict_input_assay
    print('Grouping T0 and T1 files.')
    df_T0 = concatenate_T_files(df_T0_list)
    df_T1 = concatenate_T_files(df_T1_list)
    print("Check assay types in T0.")
    validate_T0(df_T0)
    validate_T1(df_T1)
    passed, dict_assay_type = sanity_check_assay_type(df_T0, tag)
    passed_l.append(passed)
    dict_sanity["assay-types"]= dict_assay_type
    
    print("Check consistency of input_assay_id between T0 and T1.")
    passed, dict_assay_sizes = sanity_check_assay_sizes(df_T0, df_T1)
    passed_l.append(passed)
    dict_sanity["assay-sizes"]= dict_assay_sizes
    

    print("Check uniqueness of T0 and T2.")
    passed, dict_unique = sanity_check_uniqueness(df_T0, colname="input_assay_id", filename="T0")
    passed_l.append(passed)
    dict_sanity["uniqueness"]= dict_unique

    print(f"Sanity checks took {time.time() - start:.08} seconds.")
    
    dict_report["sanity_checks"]= dict_sanity
    if False in passed_l:
        save_run_report(_args, dict_report, mode="prepare_activity_data")
        exit("Found error. Please check the report.")
    else:
        print(f"Sanity checks passed.")

    start = time.time()

    print("Start aggregating values.")
    dict_aggr = {}
    output_dir_agg = aggregate_values.prepare(_args, overwriting)

    (
        df_T4r,
        df_failed_range,
        df_failed_binary,
        df_failed_aggr,
        df_failed_std,
        df_dup,
        df_T0_upd,
    ) = aggregate_values.aggregate_replicates(
        df_T0, df_T1, df_T5, ConfigDict.get_parameters()["credibility_range"], num_cpu
    )
    dict_aggr["passed"] = df_T4r.shape[0]
    dict_aggr["failed_range"] = df_failed_range.shape[0]
    dict_aggr["failed_binary"] = df_failed_binary.shape[0]
    dict_aggr["failed_aggr"] = df_failed_aggr.shape[0]
    dict_aggr["failed_std"] = df_failed_std.shape[0]
    dict_aggr["duplicates"] = df_dup.shape[0]
    dict_report["aggregation"] = dict_aggr
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
        df_failed_binary,
        "failed_binary_T1",
        ["input_compound_id", "input_assay_id",
            "standard_qualifier", "standard_value"],
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
    df_T_cat = read_input_file(_args["catalog_file"])
    dict_thres = {}
    output_dir_thres = apply_thresholding.prepare(_args, overwriting)
    df_T0_upd = df_T0_upd.astype({"input_assay_id": "int"})
    df_T4r = df_T4r.astype({"input_assay_id": "int"})
    df_T4c, df_T3c = apply_thresholding.run(df_T0_upd, df_T4r,df_T_cat,  num_cpu)
    
    # Write final dataframes (T4c, T3c)
    columns_T3c = [
        "classification_task_id",
        "input_assay_id",
        "assay_type",
        "variance_quorum_OK",
        "use_in_regression",
        "is_binary",
        "is_auxiliary",
        "threshold",
        "threshold_method",
        "direction",
        "catalog_assay_id",
        "catalog_task_id"]
    
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
    dict_thres["passed"] = df_T4c.shape[0]
    dict_thres["failed_labels"] = df_T4c_failed.shape[0]

    dict_report["thresholding"] = dict_thres
    save_df_as_csv(output_dir_thres, df_T4c_failed, "T4c.FAILED")
    save_df_as_csv(output_dir_thres, df_T4c, "T4c")
    save_df_as_csv(output_dir_thres, df_T3c, "T3c")

    print(f"Thresholding took {time.time() - start:.08} seconds.")
    print(f"Thresholding done.")

    print("Start filter classification data.")
    start = time.time()
    dict_clf_filter = {}
    output_dir_filter_clf = filter_classification.prepare(_args, overwriting)
    T10c, T8c, T4c_filtered_out, T4c_dedup = filter_classification.filter_clf(
        df_T3c,
        df_T4c,
        ConfigDict.get_parameters()["training_quorum"]["classification"],
        ConfigDict.get_parameters()["evaluation_quorum"]["classification"],
        ConfigDict.get_parameters()["training_quorum"]["CATALOG-PANEL"],
        ConfigDict.get_parameters()["initial_task_weights"],
    )
    dict_clf_filter["catalog_assays_changed_to_NON-CATALOG-PANEL_assays"] = T8c.loc[(T8c["catalog_check_INFO"] == "missing_ref_task")|(T8c["catalog_check_INFO"] == "failed_quorum"), "input_assay_id"].unique().tolist()
    counts = counts_per_type(T8c, "both",  "clf")
    dict_clf_filter["assays_tasks_per_type"] = counts.set_index("assay_type").to_dict("index")
    dict_clf_filter["dupplicates_desc_task_id"] = T4c_dedup.shape[0]
    dict_clf_filter["filtered_clf_tasks"] = T4c_filtered_out.shape[0]
    T0_cols_wo_exp_thres = [col for col in df_T0.columns if 'expert_threshold' not in col]
    T0_wo_exp_thres = df_T0[T0_cols_wo_exp_thres].copy()
    T0_wo_exp_thres.set_index("input_assay_id", inplace=True)
    T0_T8_diff_col = T0_wo_exp_thres.columns.difference(T8c.columns)
    T0_diff = T0_wo_exp_thres[T0_T8_diff_col].copy().reset_index()
    T8c = T8c.merge(T0_diff, left_on="input_assay_id", right_on="input_assay_id", how="left")
   
    filter_classification.write_tmp_output(
        output_dir_filter_clf, T10c, T8c, T4c_filtered_out, T4c_dedup
    )

    del df_T4c, df_T3c, T4c_filtered_out, T4c_dedup, df_T0

    print(f"Classification filtering took {time.time() - start:.08} seconds.")
    print(f"Classification filtering done.")
    print("Start filter regression data.")
    #####
    start = time.time()
    out_dir_filter_reg = filter_regression.prepare(_args, overwriting)
    dict_reg_filter = {}
    T10r, T8r, T4r_filtered_out, T4r_dedup = filter_regression.filter_regression_tasks(
        df_T0_upd,
        df_T4r,
        ConfigDict.get_parameters()["training_quorum"]["regression"],
        ConfigDict.get_parameters()["evaluation_quorum"]["regression"],
        ConfigDict.get_parameters()["initial_task_weights"],
        ConfigDict.get_parameters()["censored_downweighting"],
    )
    dict_reg_filter["dupplicates_desc_task_id"] = T4r_dedup.shape[0]
    dict_reg_filter["filtered_reg_data"] = T4r_filtered_out.shape[0]
    
    T8r = filter_regression.filter_regression_evaluation_tasks(
        T8r,
        T10r,
        ConfigDict.get_parameters()["regression_evaluation_task_filter"],)
    filter_regression.write_tmp_output(
        out_dir_filter_reg, T10r, T8r, T4r_filtered_out, T4r_dedup
    )
    del df_T4r, T4r_filtered_out, T4r_dedup
    print(f"Filtering regression data took {time.time() - start:.08} seconds.")
    print(f"Filtering regression data done.")

    print("Start creating sparse matrices.")

    start = time.time()
    out_dir_matrices, results_dir = csv_2_mtx.prepare(_args, overwriting)

    df_T6_cont, T10c_cont, T10r_cont = csv_2_mtx.get_cont_id(df_T6, T10c, T10r)
    df_T11 = df_T6_cont[["cont_descriptor_vector_id", "fold_id", "fp_feat"]]
    dict_clf_filter["cls_datapoints"] = T10c_cont.shape[0]
    dict_clf_filter["cls_tasks"] = T10c_cont["cont_classification_task_id"].nunique()

    dict_reg_filter["reg_datapoints"] = T10r_cont.shape[0]
    dict_reg_filter["censored_reg_datapoints"] = T10r_cont[T10r_cont["is_uncensored"] == False].shape[0]
    dict_reg_filter["reg_tasks"] = T10r_cont["cont_regression_task_id"].nunique()
    
    dict_report["filter_clf"] = dict_clf_filter
    dict_report["filter_reg"] = dict_reg_filter
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
    dict_matrices = {}
    del df_T11, T10c_cont, T10r_cont
    y_matrix_clf.data = np.nan_to_num(y_matrix_clf.data, copy=False)
    y_matrix_clf.eliminate_zeros()
    dict_matrices["x_matrix_feature_dim"] = x_matrix.shape[1]
    dict_matrices["x_matrix_compounds"] =  x_matrix.shape[0]
    dict_matrices["y_matrix_clf_values"] = y_matrix_clf.count_nonzero()
    dict_matrices["y_matrix_reg_values"] = y_matrix_reg.count_nonzero()
    dict_matrices["censored_values"] = censored_mask.count_nonzero()
    dict_matrices["y_matrix_clf_tasks"] = y_matrix_clf.shape[1]
    dict_matrices["y_matrix_reg_tasks"] = y_matrix_reg.shape[1]
    dict_report["matrix_dimensions"] = dict_matrices
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
    run_time = end - start_total
    dict_overall = {}
    dict_overall["run_time_overall"] = run_time
    dict_overall["fraction_censored_reg"] = np.round((dict_matrices["censored_values"]/dict_matrices["y_matrix_reg_values"])*100, 2)
    dict_report["overall_checks"] = dict_overall
    dict_report["run_time_overall"]  = run_time
    print(f"Overall processing took {run_time:.08} seconds.")
    save_run_report(_args, dict_report, mode="prepare_activity_data")
    print(f"Files are ready for SparseChem.")


prepare_activity_data.set_defaults(func=do_prepare_activity_data)






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


def do_prepare_prediction_online(input_structure: DataFrame, key_path: str, config_file: str, num_cpu: int)-> Tuple[csv_2_mtx.csr_matrix, DataFrame, DataFrame]:
    """Function to run the entire pipeline for prediction keeping objects in memory

    Args:
        args (Namespace): Subparser arguments
    """
    _args = {}
    _args["config_file"] = config_file
    _args["key_file"] = key_path
    num_cpu = num_cpu
    load_config(_args)
    load_key(_args)
    bit_size = config.parameters.get_parameters()["fingerprint"]["fold_size"]
    df_failed = DataFrame(columns=["input_compound_id", "error_message"])
    # TODO: Consistency check without writing to files

    # TODO: Output a single DataFrame instead of one for success and one for failures ?
    dt_std = standardize_smiles.prepare_data_transformer(num_cpu)
    df_smi, df_failed_smi = standardize_smiles.run(input_structure, dt_std)
    if df_failed_smi is not None:
        df_failed = df_failed_smi[["input_compound_id", "error_message"]]

    dt_desc = calculate_descriptors.prepare_data_transformer(num_cpu)
    df_desc, df_failed_desc = calculate_descriptors.run(df_smi, dt_desc)
    if df_failed_desc is not None:
        df_failed = pd.concat([df_failed, df_failed_desc[["input_compound_id", "error_message"]]])

    df_desc_c = df_desc.copy()
    df_desc_c.loc[:, "descriptor_vector_id"] = (
        df_desc_c.groupby("input_compound_id").ngroup().replace(-1, np.nan).add(1)
    )
    df_T6 = df_desc_c[["descriptor_vector_id", "fp_feat", "fp_val"]]
    df_T11 = map_2_cont_id(df_T6, "descriptor_vector_id").sort_values(
        "cont_descriptor_vector_id"
    )
    mapping_table = df_desc_c[["input_compound_id", "descriptor_vector_id"]]
    mapping_table = mapping_table.merge(df_T11[["descriptor_vector_id", "cont_descriptor_vector_id"]], on="descriptor_vector_id", how="left").sort_values("cont_descriptor_vector_id")
    #TODO: Return the df_T11 DataFrame instead of the csr matrix, or both ?
    x_matrix = csv_2_mtx.matrix_from_strucutres(df_T11, bit_size)
    return x_matrix, df_failed, mapping_table



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
    bit_size = config.parameters.get_parameters()["fingerprint"][
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
