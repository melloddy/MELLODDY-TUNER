import argparse
from argparse import Namespace
from typing import Tuple
import pandas as pd
from pandas.core.frame import DataFrame
import melloddy_tuner
from melloddy_tuner.utils import hash_reference_set
from melloddy_tuner.utils.formatting import ActivityDataFormatting
from melloddy_tuner.utils.helper import (
    create_log_files,
    load_config,
    load_key,
    make_dir,
    read_input_file,
    save_df_as_csv,
    save_mtx_as_npy,
    save_run_report,
)


import time
from itertools import chain
from pathlib import Path

import melloddy_tuner as tuner
import numpy as np
from scipy.io import mmwrite
from scipy.sparse import csr_matrix


def init_arg_parser() -> Namespace:
    parser = argparse.ArgumentParser(description="Create sparse matrices.")
    parser.add_argument(
        "-s",
        "--structure_file",
        type=str,
        help="path of the processed structure input file T6",
        required=True,
    )
    parser.add_argument(
        "-ac",
        "--activity_file_clf",
        type=str,
        help="path of the processed classification activity file T10c",
    )
    parser.add_argument(
        "-wc",
        "--weight_table_clf",
        type=str,
        help="path of the processed classification weight table file T8c",
    )
    parser.add_argument(
        "-ar",
        "--activity_file_reg",
        type=str,
        help="path of the processed regression activity file T10r",
    )
    parser.add_argument(
        "-wr",
        "--weight_table_reg",
        type=str,
        help="path of the processed regression weight table file T8r",
    )
    parser.add_argument(
        "-c", "--config_file", type=str, help="path of the config file", required=True
    )
    parser.add_argument(
        "-k", "--key_file", type=str, help="path of the key file", required=True
    )

    parser.add_argument(
        "-o", "--output_dir", type=str, help="path to output directory", required=True
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
        "-aux",
        "--using_aux",
        choices=["no", "yes"],
        help="tag to identify if auxiliary data is used. Available tags: no or yes",
        required=True,
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


def matrix_from_strucutres(df: DataFrame, bit_size: int) -> csr_matrix:
    """
    Create sparse matrix in csr format from dataframe by mapping it to certain bitsize.

    Args:
        df (DataFrame): structure dataframe T11
        bit_size (int): given bitsize from config

    Returns:
        csr_matrix: structure matrix X
    """
    # create a dictionary mapping ecfp hash keys to column indices in X.mtx used for training
    # ofc, the file here should be passed as a new argument

    df["fp_feat"] = df["fp_feat"].str[1:-1]
    cols = df.columns.difference(["fp_feat"])
    bits = df.fp_feat.str.split(",")
    x_ijv_df = df.loc[df.index.repeat(bits.str.len()), cols].assign(
        bits=list(chain.from_iterable(bits.tolist()))
    )
    x_ijv_df["bits"] = x_ijv_df["bits"].astype(np.int32)
    # works only for binary here: but could pick up the values in column fp_val
    x_ijv_df["value"] = np.ones(x_ijv_df.shape[0]).astype(np.int8)
    data = x_ijv_df["value"].values

    # get the row coordinates of the X matrix
    I, J = x_ijv_df["cont_descriptor_vector_id"], x_ijv_df["bits"]

    # create the matrix, make sure it has the right dimension: here [number molecules to predict x number  defined by the parameter 'fold_size']
    matrix = csr_matrix((data.astype(np.int8), (I, J)), shape=(df.shape[0], bit_size))

    return matrix


def create_csr_matrix(df, ids, map_rows, map_cols, label, task="classification"):
    I, J = (
        [map_rows[x] for x in df["cont_descriptor_vector_id"]],
        [map_cols[x] for x in df[f"cont_{task}_task_id"]],
    )
    return csr_matrix(
        (df[label].values.astype(np.float32), (I, J)),
        shape=(len(ids), len(map_cols)),
    )


def get_censored_mask(regression_data, map_rows, map_cols_regr):
    """
    Retrieve the censored mask from the regression_data
    """

    relation_col = "standard_qualifier"
    lower_censored = ["<"]
    upper_censored = [">"]
    uncensored = ["="]
    reference_set = lower_censored + uncensored + upper_censored
    relation_set = regression_data[relation_col].unique()
    for r in relation_set:
        if r not in reference_set:
            print(f"{relation_set} contains element not present in {reference_set}")
            quit()
    # build sparse matrix
    df_lower = regression_data[regression_data[relation_col].isin(lower_censored)]
    I_lower, J_lower = (
        [map_rows[x] for x in df_lower["cont_descriptor_vector_id"]],
        [map_cols_regr[x] for x in df_lower[f"cont_regression_task_id"]],
    )

    df_censored = regression_data[regression_data[relation_col].isin(uncensored)]
    I_censored, J_censored = (
        [map_rows[x] for x in df_censored["cont_descriptor_vector_id"]],
        [map_cols_regr[x] for x in df_censored[f"cont_regression_task_id"]],
    )

    df_upper = regression_data[regression_data[relation_col].isin(upper_censored)]
    I_upper, J_upper = (
        [map_rows[x] for x in df_upper["cont_descriptor_vector_id"]],
        [map_cols_regr[x] for x in df_upper[f"cont_regression_task_id"]],
    )

    return csr_matrix(
        (
            [-1] * len(I_lower) + [0] * len(I_censored) + [1] * len(I_upper),
            (I_lower + I_censored + I_upper, J_lower + J_censored + J_upper),
        ),
        shape=(len(map_rows), len(map_cols_regr)),
    )


# def matrix_from_activity(df_clf: DataFrame, df_reg: DataFrame) -> Tuple:

#     desc_ids_used = np.unique(
#         df_clf['cont_descriptor_vector_id'].append(df_reg['cont_descriptor_vector_id'])
#     )
#     map_rows = {val: ind for ind, val in enumerate(desc_ids_used)}

#     map_cols = {
#         val: ind for ind, val in enumerate(np.unique(df_clf['cont_classification_task_id']))
#     }
#     map_cols_regr = {
#         val: ind
#         for ind, val in enumerate(np.unique(df_reg['cont_regression_task_id']))
#     }

#     if len(df_clf) > 0:
#         matrix_class = create_csr_matrix(
#             df_clf, desc_ids_used, map_rows, map_cols, 'class_label', task='classification'
#         )
#     else:
#         matrix_class = None

#     if len(df_reg) > 0:
#         matrix_regr = create_csr_matrix(
#             df_reg,
#             desc_ids_used,
#             map_rows,
#             map_cols_regr,
#             'standard_value',
#             task='regression',
#         )
#         censored_mask = get_censored_mask(df_reg, map_rows, map_cols_regr)
#     else:
#         matrix_regr = None
#         censored_mask = None
#     return matrix_class, matrix_regr, censored_mask


def matrix_from_activity(df, df_regr):
    desc_ids_used = np.unique(
        df["cont_descriptor_vector_id"].append(df_regr["cont_descriptor_vector_id"])
    )
    map_rows = {val: ind for ind, val in enumerate(desc_ids_used)}

    map_cols = {
        val: ind for ind, val in enumerate(np.unique(df["cont_classification_task_id"]))
    }
    map_cols_regr = {
        val: ind
        for ind, val in enumerate(np.unique(df_regr["cont_regression_task_id"]))
    }

    if len(df) > 0:
        matrix_class = create_csr_matrix(
            df, desc_ids_used, map_rows, map_cols, "class_label"
        )
    else:
        matrix_class = None

    if len(df_regr) > 0:
        matrix_regr = create_csr_matrix(
            df_regr,
            desc_ids_used,
            map_rows,
            map_cols_regr,
            "standard_value",
            task="regression",
        )
        censored_mask = get_censored_mask(df_regr, map_rows, map_cols_regr)
    else:
        matrix_regr = None
        censored_mask = None

    return matrix_class, matrix_regr, censored_mask


def folding_from_structure(df: DataFrame) -> np.array:
    """
    Get fold ids as numpy array from T11 dataframe.

    Args:
        df (DataFrame): structure dataframe T11

    Returns:
        np.array: folding vector
    """
    folding_vector = df["fold_id"].values
    return folding_vector


def prepare(args: dict, overwriting: bool) -> Path:
    """
    Create output folder for matrices

    Args:
        args (dict): argparser dictionary
        overwriting (bool): overwriting argument

    Returns:
        Path: output path
    """
    output_dir = make_dir(args, "matrices", None, overwriting)
    results_dir = make_dir(args, "results", None, overwriting)

    create_log_files(output_dir)
    return output_dir, results_dir


def make_matrices(
    df_T11: DataFrame, df_T10c: DataFrame, df_T10r: DataFrame, bit_size: int
) -> Tuple:
    """
    Wrapper to create X and Y matrices and fold vector.

    Args:
        df_T11 (DataFrame): structure dataframe T11
        df_T10 (DataFrame): activity dataframe T10
        bit_size (int): bitsize from config

    Returns:
        Tuple: X matrix, Y matrix, fold vector
    """
    x_matrix = matrix_from_strucutres(df_T11, bit_size)
    # return x_matrix
    fold_vector = folding_from_structure(df_T11)
    y_matrix_clf, y_matrix_reg, y_censored_mask = matrix_from_activity(df_T10c, df_T10r)

    return x_matrix, fold_vector, y_matrix_clf, y_matrix_reg, y_censored_mask


def save_csv_output(
    out_dir: Path, tag: str, df_T9c: DataFrame, df_T9r: DataFrame
) -> None:
    """
    Wrapper to save csv files (counts.csv and weights.csv) to matrix output folder.

    Args:
        out_dir (Path): path to matrix output folder
        df_T10_counts (DataFrame): activity dataframe T10_counts containing counts per task.
        df_T3_mapped (DataFrame): Mapped weight tabel T3
    """

    df_T9c = df_T9c.rename(
        columns={
            ("cont_classification_task_id"): "task_id",
            ("catalog_task_id"): "catalog_id",
            ("assay_type"): "task_type",
            ("weight"): "training_weight",
        }
    )
    df_T9c = df_T9c.dropna(subset=["task_id"]).sort_values("task_id")
    df_T9c.loc[:,"task_id"] = df_T9c["task_id"].astype(int)
    df_T9r = df_T9r.rename(
        columns={
            ("cont_regression_task_id"): "task_id",
            ("assay_type"): "task_type",
            ("weight"): "training_weight",
        }
    )
    df_T9r = df_T9r.dropna(subset=["task_id"]).sort_values("task_id")
    df_T9r.loc[:,"task_id"] = df_T9r["task_id"].astype(int)
    if tag == "no":
        out_dir_wo = out_dir / "wo_aux"
        out_dir_wo.mkdir(exist_ok=True)

        out_dir_cls = out_dir_wo / "cls"
        out_dir_cls.mkdir(exist_ok=True)
        save_df_as_csv(
            out_dir_cls,
            df_T9c,
            "cls_weights",
            ["task_id", "catalog_id", "task_type", "training_weight", "aggregation_weight"],
        )
        out_dir_reg = out_dir_wo / "reg"
        out_dir_reg.mkdir(exist_ok=True)
        save_df_as_csv(
            out_dir_reg,
            df_T9r,
            "reg_weights",
            [
                "task_id",
                "task_type",
                "training_weight",
                "aggregation_weight",
                "censored_weight",
            ],
        )
        out_dir_hybrid = out_dir_wo / "hyb"
        out_dir_hybrid.mkdir(exist_ok=True)
        save_df_as_csv(
            out_dir_hybrid,
            df_T9c,
            "hyb_cls_weights",
            ["task_id", "catalog_id", "task_type", "training_weight", "aggregation_weight"],
        )
        save_df_as_csv(
            out_dir_hybrid,
            df_T9r,
            "hyb_reg_weights",
            [
                "task_id",
                "task_type",
                "training_weight",
                "aggregation_weight",
                "censored_weight",
            ],
        )
    if tag == "yes":
        out_dir_w = out_dir / "w_aux"
        out_dir_w.mkdir(exist_ok=True)
        out_dir_clsaux = out_dir_w / "clsaux"
        out_dir_clsaux.mkdir(exist_ok=True)
        save_df_as_csv(
            out_dir_clsaux,
            df_T9c,
            "clsaux_weights",
            ["task_id","catalog_id", "task_type", "training_weight", "aggregation_weight"],
        )
        out_dir_hybrid = out_dir_w / "hybrid"
        out_dir_hybrid.mkdir(exist_ok=True)
        save_df_as_csv(
            out_dir_hybrid,
            df_T9c,
            "hyb_cls_weights",
            ["task_id", "catalog_id", "task_type", "training_weight", "aggregation_weight"],
        )
        save_df_as_csv(
            out_dir_hybrid,
            df_T9r,
            "hyb_reg_weights",
            [
                "task_id",
                "task_type",
                "training_weight",
                "aggregation_weight",
                "censored_weight",
            ],
        )


def save_npy_matrices(
    out_dir: Path,
    tag: str,
    x_matrix: csr_matrix,
    fold_vector: np.array,
    y_matrix_clf: csr_matrix,
    y_matrix_reg: csr_matrix,
    censored_mask: csr_matrix,
) -> None:
    """
    Wrapper to save csr files to matrix output folder.

    Args:
        out_dir (Path): path to matrix output folder.
        x_matrix (csr_matrix): csr structure matrix X
        y_matrix (csr_matrix): csr activity matrix Y
        fold_vector (np.array): fold vector
    """
    if tag == "no":
        out_dir_wo = out_dir / "wo_aux"
        out_dir_wo.mkdir(exist_ok=True)
        out_dir_cls = out_dir_wo / "cls"
        out_dir_cls.mkdir(exist_ok=True)
        save_mtx_as_npy(x_matrix, out_dir_cls, f"cls_T11_x")
        save_mtx_as_npy(fold_vector, out_dir_cls, f"cls_T11_fold_vector")
        save_mtx_as_npy(y_matrix_clf, out_dir_cls, f"cls_T10_y")
        out_dir_reg = out_dir_wo / "reg"
        out_dir_reg.mkdir(exist_ok=True)
        save_mtx_as_npy(x_matrix, out_dir_reg, "reg_T11_x")
        save_mtx_as_npy(fold_vector, out_dir_reg, "reg_T11_fold_vector")
        save_mtx_as_npy(y_matrix_reg, out_dir_reg, "reg_T10_y")
        save_mtx_as_npy(censored_mask, out_dir_reg, "reg_T10_censor_y")

        out_dir_hyb = out_dir_wo / "hyb"
        out_dir_hyb.mkdir(exist_ok=True)
        save_mtx_as_npy(x_matrix, out_dir_hyb, f"hyb_T11_x")
        save_mtx_as_npy(fold_vector, out_dir_hyb, f"hyb_T11_fold_vector")
        save_mtx_as_npy(y_matrix_clf, out_dir_hyb, f"hyb_cls_T10_y")
        save_mtx_as_npy(y_matrix_reg, out_dir_hyb, "hyb_reg_T10_y")
        save_mtx_as_npy(censored_mask, out_dir_hyb, "hyb_reg_T10_censor_y")

    if tag == "yes":
        out_dir_w = out_dir / "w_aux"
        out_dir_w.mkdir(exist_ok=True)
        out_dir_clsaux = out_dir_w / "clsaux"
        out_dir_clsaux.mkdir(exist_ok=True)
        save_mtx_as_npy(x_matrix, out_dir_clsaux, f"clsaux_T11_x")
        save_mtx_as_npy(fold_vector, out_dir_clsaux, f"clsaux_T11_fold_vector")
        save_mtx_as_npy(y_matrix_clf, out_dir_clsaux, f"clsaux_T10_y")

        # out_dir_hyb = out_dir_w / "hybrid"
        # out_dir_hyb.mkdir(exist_ok=True)
        # save_mtx_as_npy(x_matrix, out_dir_hyb, f"hyb_T11_x")
        # save_mtx_as_npy(fold_vector, out_dir_hyb, f"hyb_T11_fold_vector")
        # save_mtx_as_npy(y_matrix_clf, out_dir_hyb, f"hyb_cls_T10_y")
        # save_mtx_as_npy(y_matrix_reg, out_dir_hyb, "hyb_reg_T10_y")
        # save_mtx_as_npy(censored_mask, out_dir_hyb, "hyb_reg_T10_censor_y")





def map_2_cont_id(data: pd.DataFrame, column_name: str):
    map_id = {val: ind for ind, val in enumerate(np.unique(data[column_name]))}
    map_id_df = pd.DataFrame.from_dict(map_id, orient="index").reset_index()
    map_id_df = map_id_df.rename(
        columns={"index": column_name, 0: "cont_" + column_name}
    )
    data_remapped = pd.merge(data, map_id_df, how="inner", on=column_name)

    return data_remapped


def get_cont_id(T6, T10c, T10r):
    map_id = {
        val: ind
        for ind, val in enumerate(
            np.unique(T10c["descriptor_vector_id"].append(T10r["descriptor_vector_id"]))
        )
    }
    map_id_df = (
        pd.DataFrame.from_dict(map_id, orient="index")
        .reset_index()
        .rename(
            columns={
                "index": "descriptor_vector_id",
                0: "cont_" + "descriptor_vector_id",
            }
        )
    )
    T10c_cont = pd.merge(
        T10c,
        map_id_df,
        how="inner",
        on="descriptor_vector_id",
    ).sort_values("cont_classification_task_id")
    T10r_cont = pd.merge(
        T10r,
        map_id_df,
        how="inner",
        on="descriptor_vector_id",
    ).sort_values("cont_regression_task_id")

    data_mapped_desc = map_id_df.copy()
    T6_filtered = T6[
        T6.descriptor_vector_id.isin(data_mapped_desc.descriptor_vector_id)
    ]
    data_mapped_desc.set_index("descriptor_vector_id", inplace=True)
    T6_filtered.set_index("descriptor_vector_id", inplace=True)
    T6_cont = T6_filtered.join(
        data_mapped_desc, on="descriptor_vector_id"
    ).drop_duplicates()
    T6_cont = T6_cont.reset_index().sort_values("cont_descriptor_vector_id")
    return T6_cont, T10c_cont, T10r_cont


def main(args: dict = None):
    """
    Main function reading input files, executing functions and writing output files.
    """
    start = time.time()
    dict_report = {}
    dict_matrices = {}
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
    dict_report["run_parameters"] =  args
    print("Generate sparse matrices from given dataframes.")
    fp_param = melloddy_tuner.utils.config.parameters.get_parameters()["fingerprint"]
    bit_size = fp_param["fold_size"]
    output_dir, results_dir = prepare(args, overwriting)
    tag = args["using_auxiliary"]

    if (tag != "no") and (tag != "yes"):
        print("Please choose a different tag. Only no or yes are allowed.")
        exit()
    df_T6 = read_input_file(args["structure_file"])
    df_T10c = read_input_file(args["activity_file_clf"])
    df_T10r = read_input_file(args["activity_file_reg"])
    df_T6_cont, T10c_cont, T10r_cont = get_cont_id(df_T6, df_T10c, df_T10r)
    df_T11 = df_T6_cont[["cont_descriptor_vector_id", "fold_id", "fp_feat"]]

    df_T9c = read_input_file(args["weight_table_clf"])
    df_T9r = read_input_file(args["weight_table_reg"])

    save_df_as_csv(results_dir, T10c_cont, "T10c_cont")
    save_df_as_csv(results_dir, T10r_cont, "T10r_cont")
    save_df_as_csv(results_dir, df_T6_cont, "T6_cont")

    save_csv_output(output_dir, tag, df_T9c, df_T9r)

    x_matrix, fold_vector, y_matrix_clf, y_matrix_reg, censored_mask = make_matrices(
        df_T11, T10c_cont, T10r_cont, bit_size
    )
    y_matrix_clf.data = np.nan_to_num(y_matrix_clf.data, copy=False)
    y_matrix_clf.eliminate_zeros()
    dict_matrices["x_matrix_feature_dim"] = x_matrix.shape[1]
    dict_matrices["x_matrix_compounds"] =  x_matrix.shape[0]
    dict_matrices["y_matrix_clf_values"] = y_matrix_clf.count_nonzero()
    dict_matrices["y_matrix_reg_values"] = y_matrix_reg.count_nonzero()
    dict_matrices["censored_values"] = censored_mask.count_nonzero()
    dict_matrices["y_matrix_clf_tasks"] = y_matrix_clf.shape[1]
    dict_matrices["y_matrix_reg_tasks"] = y_matrix_reg.shape[1]

    dict_matrices["catalog_tasks"] =  df_T9c.catalog_task_id.nunique()
    save_npy_matrices(
        output_dir,
        tag,
        x_matrix,
        fold_vector,
        y_matrix_clf,
        y_matrix_reg,
        censored_mask,
    )
    end = time.time()
    run_time = end - start
    dict_report["sparse_matrices"] = dict_matrices
    dict_report["run_time"] = run_time
    save_run_report(args, dict_report,"sparse_matrices")
    print(f"Formatting to matrices took {run_time:.08} seconds.")
    print(f"Files are ready for SparseChem.")


if __name__ == "__main__":
    main()
