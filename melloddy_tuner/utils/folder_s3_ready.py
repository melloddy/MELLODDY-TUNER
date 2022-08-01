import argparse

import sys, time, os
from pathlib import Path
import shutil
import typing
from melloddy_tuner.utils.helper import (
    save_run_args,
    read_run_params

)

CLASSIFICATION_DATASET_NAMES = {
    "X_MATRIX_NAME": "cls_T11_x.npz",
    "FOLD_VECTOR_NAME": "cls_T11_fold_vector.npy",
    "CLS_Y_MATRIX_NAME": "cls_T10_y.npz",
    "CLS_WEIGHT_NAME": "cls_weights.csv",
}
REGRESSION_DATASET_NAMES = {
    "X_MATRIX_NAME": "reg_T11_x.npz",
    "FOLD_VECTOR_NAME": "reg_T11_fold_vector.npy",
    "REG_Y_MATRIX_NAME": "reg_T10_y.npz",
    "REG_WEIGHT_NAME": "reg_weights.csv",
    "Y_CENSORING_MATRIX_NAME": "reg_T10_censor_y.npz",
}
CLASSIFICATION_AUX_DATASET_NAMES = {
    "X_MATRIX_NAME": "clsaux_T11_x.npz",
    "FOLD_VECTOR_NAME": "clsaux_T11_fold_vector.npy",
    "CLS_Y_MATRIX_NAME": "clsaux_T10_y.npz",
    "CLS_WEIGHT_NAME": "clsaux_weights.csv",
}

HYBRID_DATASET_NAMES = {
    "X_MATRIX_NAME": "hyb_T11_x.npz",
    "CLS_Y_MATRIX_NAME": "hyb_cls_T10_y.npz",
    "REG_Y_MATRIX_NAME": "hyb_reg_T10_y.npz",
    "FOLD_VECTOR_NAME": "hyb_T11_fold_vector.npy",
    "Y_CENSORING_MATRIX_NAME": "hyb_reg_T10_censor_y.npz",
    "CLS_WEIGHT_NAME": "hyb_cls_weights.csv",
    "REG_WEIGHT_NAME": "hyb_reg_weights.csv",
}

def init_arg_parser():
    """Argparser module to load commandline arguments.

    Returns:
        [Namespace]: Arguments from argparser
    """
    make_folders_s3 = argparse.ArgumentParser(description="create folder structure for S3 bucket")
    
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
    args = make_folders_s3.parse_args()
    return args


def validate_filenames(dataset_type: typing.Optional[str], asset_filepaths):
    """ FROM MELLOFLOW
    Throw if filenames are not matching an expected dataset type
    """
    if dataset_type is None:
        raise Exception(
            f"Could not get the dataset type. Downloaded files: {asset_filepaths}."
        )
    else:
        if dataset_type == "hyb":
            mandatory_val = HYBRID_DATASET_NAMES.values()
        elif dataset_type == "cls":
            mandatory_val = CLASSIFICATION_DATASET_NAMES.values()
        elif dataset_type == "clsaux":
            mandatory_val = CLASSIFICATION_AUX_DATASET_NAMES.values()
        elif dataset_type == "reg":
            mandatory_val = REGRESSION_DATASET_NAMES.values()

        # Assert that we have all the necessary files
        missing = [k for k in mandatory_val if k not in asset_filepaths]
        if missing:
            raise Exception(f"Missing files for {dataset_type} dataset: {str(missing)}")
         # Assert that we have all the necessary files
         
        additional = [k for k in asset_filepaths if k not in mandatory_val]
        if additional:
            raise Exception(f"Additional files for {dataset_type} dataset: {str(additional)}")

def main(args: dict = None):
    #######################################
    """
    Make Folders for Transfer to S3
    """

    

    start = time.time()
    dict_report = {}
    if args is None:
        args = vars(init_arg_parser())
    _args = args
    
    
        
    dict_report["run_parameters"] = _args
    



 
    #########
    phase_list = ["phase1", "phase2", "phase3"]
    dataset_types = ["cls", "reg", "clsaux", "hyb"]
    start = time.time()
    tag = _args["using_auxiliary"]
    matrices_path = os.path.join(_args["output_dir"], _args["run_name"], "matrices/")

    out_dir_s3 = Path( os.path.join(_args["output_dir"], _args["run_name"], "s3_ready/"))
    if tag == "no":
        dataset_types = ["cls", "reg", "hyb"]
        for dataset_type in dataset_types:
            for phase in phase_list:

                out_dir_tmp = out_dir_s3 / f"{dataset_type}_{phase}"
                out_dir_tmp.mkdir( parents=True, exist_ok=True)
                src_dir = os.path.join(matrices_path + f"wo_aux/{dataset_type}")
                fnames = os.listdir(src_dir)
                validate_filenames(dataset_type, fnames)
                shutil.copytree(src_dir, out_dir_tmp, dirs_exist_ok=True)
                
            print(f"Files copied for dataset type: {dataset_type}")

    if tag == "yes":
        dataset_types = ["clsaux"]
        for dataset_type in dataset_types:
            for phase in phase_list:
                out_dir_tmp = out_dir_s3 / f"{dataset_type}_{phase}"
                out_dir_tmp.mkdir( parents=True, exist_ok=True)
                src_dir = os.path.join(matrices_path + f"w_aux/{dataset_type}")
                fnames = os.listdir(src_dir)
                validate_filenames(dataset_type, fnames)
                shutil.copytree(src_dir, out_dir_tmp, dirs_exist_ok=True)
                
            print(f"Files copied for dataset type: {dataset_type}")

    run_time = time.time()-start
    print(f"Copying files took {run_time:.08} seconds.")

if __name__ == "__main__":
    main()