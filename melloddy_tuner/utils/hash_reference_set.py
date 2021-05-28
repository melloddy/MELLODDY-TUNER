import argparse
from argparse import Namespace
from melloddy_tuner.utils import helper
from melloddy_tuner.utils.descriptor_calculator import DescriptorCalculator
from melloddy_tuner.utils.scaffold_folding import ScaffoldFoldAssign
from melloddy_tuner.utils.df_transformer import DfTransformer
from melloddy_tuner.utils.standardizer import Standardizer
from melloddy_tuner.utils.config import ConfigDict, SecretDict
import os
import hashlib
import json
import time
from pathlib import Path

from melloddy_tuner.utils.version import __version__
from melloddy_tuner.utils.helper import (
    load_config,
    load_key,
    make_dir,
    map_2_cont_id,
    read_input_file,
    save_df_as_csv,
)
from melloddy_tuner.scripts import standardize_smiles, calculate_descriptors


def init_arg_parser() -> Namespace:

    parser = argparse.ArgumentParser(description="Run data processing")
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
        "-n",
        "--number_cpu",
        type=int,
        help="number of CPUs for calculation (default: 1 CPU core)",
        default=1,
    )
    parser.add_argument(
        "-r", "--run_name", type=str, help="name of your current run", required=True
    )
    args = parser.parse_args()
    return args


def hash_reference_dir(args: dict, output_dir: Path):
    """ """
    config_file = args["config_file"]
    key_file = args["key_file"]
    ref_dir = output_dir
    sha256_hash = hashlib.sha256()
    if not ref_dir.exists():
        return print("Reference set directory does not exist.")
    try:
        filepath = ref_dir / "T11.csv"
        if filepath.exists() is True:
            print("Hashing unit test file", filepath)
            with open(filepath, "rb") as f:
                # Read file in as little chunks
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(
                        hashlib.sha256(byte_block).hexdigest().encode("utf-8")
                    )
                f.close()
        else:
            print("Reference set file T11 not found.")

        with open(config_file, "rb") as cfg_f:
            # Read file in as little chunks

            print("Hashing", config_file)
            # for byte_block in iter(lambda: cfg_f.read(4096), b""):
            #     sha256_hash.update(hashlib.sha256(
            #         byte_block).hexdigest().encode('utf-8'))
            tmp_dict = json.load(cfg_f)
            encoded_dict = json.dumps(tmp_dict, sort_keys=True, default=str).encode()
            sha256_hash.update(hashlib.sha256(encoded_dict).hexdigest().encode("utf-8"))
            cfg_f.close()
        with open(key_file, "rb") as key_f:
            # Read key file
            print("Hashing", key_file)
            tmp_dict = json.load(key_f)
            encoded_dict = json.dumps(tmp_dict, sort_keys=True, default=str).encode()
            sha256_hash.update(hashlib.sha256(encoded_dict).hexdigest().encode("utf-8"))
            key_f.close()
        print(f"Hashing version: {__version__}")
        sha256_hash.update(
            hashlib.sha256(__version__.encode("utf-8")).hexdigest().encode("utf-8")
        )

    except:
        import traceback

        # Print the stack traceback
        traceback.print_exc()
        return print("General Error.")
    hash_hex = sha256_hash.hexdigest()
    reference_hash = {"unit_test_hash": hash_hex}
    p_output_dir = ref_dir
    path_gen_hash = p_output_dir / ""
    path_gen_hash.mkdir(exist_ok=True)
    with open(path_gen_hash / "generated_hash.json", "w") as json_file:
        json.dump(reference_hash, json_file)
    return print("Done.")


def compare_hash_keys(args, output_dir):
    """ """
    ref_dir = output_dir
    if args["ref_hash"] is None:
        return print(
            "No reference hash given. Comparison of generated and reference hash keys will be skipped."
        )
    else:
        with open(args["ref_hash"]) as ref_hash_f:
            ref_hash = json.load(ref_hash_f)
        key_ref = ref_hash["unit_test_hash"]
        path_gen_hash = ref_dir / "generated_hash.json"
        with open(path_gen_hash) as hash_f:
            key = json.load(hash_f)
        if key["unit_test_hash"] != key_ref:
            print(
                "Different reference key. Please check the parameters you used for structure preparation."
            )
            return quit()
        else:
            return print("Identical hash keys. Continue with data processing.")


def prepare(args):
    overwriting = True

    load_config(args)
    load_key(args)
    output_dir = make_dir(args, "reference_set", None, overwriting)
    key = SecretDict.get_secrets()["key"]
    method_params_standardizer = ConfigDict.get_parameters()["standardization"]
    st = Standardizer.from_param_dict(
        method_param_dict=method_params_standardizer, verbosity=0
    )
    outcols_st = ["canonical_smiles", "success", "error_message"]
    out_types_st = ["object", "bool", "object"]
    dt_standarizer = DfTransformer(
        st,
        input_columns={"smiles": "smiles"},
        output_columns=outcols_st,
        output_types=out_types_st,
        success_column="success",
        nproc=1,
        verbosity=0,
    )

    method_params_folding = ConfigDict.get_parameters()["scaffold_folding"]
    sa = ScaffoldFoldAssign.from_param_dict(
        secret=key, method_param_dict=method_params_folding, verbosity=0
    )
    outcols_sa = ["murcko_smiles", "sn_smiles", "fold_id", "success", "error_message"]
    out_types_sa = ["object", "object", "int", "bool", "object"]
    dt_fold = DfTransformer(
        sa,
        input_columns={"canonical_smiles": "smiles"},
        output_columns=outcols_sa,
        output_types=out_types_sa,
        success_column="success",
        nproc=1,
        verbosity=0,
    )

    method_params_descriptor = ConfigDict.get_parameters()["fingerprint"]
    dc = DescriptorCalculator.from_param_dict(
        secret=key, method_param_dict=method_params_descriptor, verbosity=0
    )
    outcols_dc = ["fp_feat", "fp_val", "success", "error_message"]
    out_types_dc = ["object", "object", "bool", "object"]
    dt_descriptor = DfTransformer(
        dc,
        input_columns={"canonical_smiles": "smiles"},
        output_columns=outcols_dc,
        output_types=out_types_dc,
        success_column="success",
        nproc=1,
        verbosity=0,
    )

    return output_dir, dt_standarizer, dt_fold, dt_descriptor


def write_output(output_dir: str, dict_df: dict):
    for key in dict_df.keys():
        file_path = os.path.join(output_dir, f"{key}.csv")
        dict_df[key].to_csv(file_path, index=False)


def main(args: dict = None):
    start = time.time()
    if args is None:
        args = vars(init_arg_parser())
    if ("reference_set" not in args.keys()) or (args.get("reference_set") is None):
        print("Default reference files from data/reference_set.csv loaded.")
        main_location = os.path.dirname(os.path.realpath(__file__))
        default_reference_file = os.path.join(
            main_location, "../data/reference_set.csv"
        )
        path_structure_file = default_reference_file
    else:
        path_structure_file = args["reference_set"]
    output_dir, dt_standarizer, dt_fold, dt_descriptor = prepare(args)
    df = read_input_file(path_structure_file)
    ref_smi, ref_smi_failed = dt_standarizer.process_dataframe(df)
    ref_desc, ref_desc_failed = dt_descriptor.process_dataframe(ref_smi)
    ref_fold, ref_fold_failed = dt_fold.process_dataframe(ref_desc)
    ref_T5, ref_T6, ref_desc_dupl = helper.format_dataframe(ref_fold)
    ref_T11 = map_2_cont_id(ref_T6, "descriptor_vector_id").sort_values(
        "cont_descriptor_vector_id"
    )
    dict_df = {
        "T2_standardized": ref_smi,
        "T2_standardized.FAILED": ref_smi_failed,
        "T2_folds": ref_fold,
        "T2_folds.FAILED": ref_fold_failed,
        "T2_descriptors": ref_desc,
        "T2_desciptors.FAILED": ref_desc_failed,
        "T2_descriptors.DUPLICATES": ref_desc_dupl,
        "T5": ref_T5,
        "T6": ref_T6,
        "T11": ref_T11,
    }
    write_output(output_dir, dict_df)
    hash_reference_dir(args, output_dir)
    compare_hash_keys(args, output_dir)
    print(f"Hashing reference data finished after {time.time() - start:.08} seconds.")


if __name__ == "__main__":
    main()
