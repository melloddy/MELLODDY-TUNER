# Data Preparation Script for the IMI Project MELLODDY <!-- omit in toc -->


- [Installation](#installation)
  - [Requirements](#requirements)
  - [Setup the environment](#setup-the-environment)
    - [Clone git repository](#clone-git-repository)
    - [Create enviroment](#create-enviroment)
  - [Package Installation](#package-installation)
- [Input and Output Files for Year 3](#input-and-output-files-for-year-3)
  - [Preparation of Input Files (version year 3)](#preparation-of-input-files-version-year-3)
    - [T0 weight table (T0) (Required column)](#t0-weight-table-t0-required-column)
    - [T1 activity file](#t1-activity-file)
    - [T2 structure file](#t2-structure-file)
  - [Expected Output Files](#expected-output-files)
- [1. Run Data Prepration Script](#1-run-data-prepration-script)
    - [`prepare_4_training`](#prepare_4_training)
  - [`prepare_structure_data`](#prepare_structure_data)
  - [`prepare_activity_data`](#prepare_activity_data)
  - [`prepare_4_prediction`](#prepare_4_prediction)
- [2 Individual scripts](#2-individual-scripts)
  - [Input file paths definition](#input-file-paths-definition)
  - [2.1  `standardize_smiles`](#21--standardize_smiles)
  - [2.2 `calculate_descriptors`](#22-calculate_descriptors)
  - [2.3.1 `assign_fold` with the scaffold-based split](#231-assign_fold-with-the-scaffold-based-split)
  - [2.3.2 `assign_lsh_fold` with Locality Sensitive Hashing (LSH)](#232-assign_lsh_fold-with-locality-sensitive-hashing-lsh)
  - [2.4 `agg_activity_data`](#24-agg_activity_data)
  - [2.5 `apply_thresholding`](#25-apply_thresholding)
  - [2.6 classification tasks filtering](#26-classification-tasks-filtering)
  - [2.7 regression tasks filtering](#27-regression-tasks-filtering)
  - [2.8 `make_matrices`](#28-make_matrices)
  - [2.9 `make_folders_s3`](#29-make_folders_s3)
- [Parameter definitions](#parameter-definitions)
    - [standardization](#standardization)
    - [fingerprint](#fingerprint)
    - [scaffold_folding](#scaffold_folding)
    - [credibility_range](#credibility_range)
    - [train_quorum](#train_quorum)
      - [regression](#regression)
      - [classification](#classification)
    - [evaluation_quorum](#evaluation_quorum)
      - [regression](#regression-1)
      - [classification](#classification-1)
    - [initial_task_weights](#initial_task_weights)
    - [global_thresholds](#global_thresholds)
    - [censored_downweighting](#censored_downweighting)
    - [count_task](#count_task)
    - [lsh](#lsh)





# Installation

Version: **`3.0.2`**
## Requirements
The data preprocessing script requires:
1. Python 3.8 or higher
2. Local Conda installation (e.g. miniconda)
3. Git installation


## Setup the environment

### Clone git repository
First, clone the git repository from the MELLODDY gitlab repository:
```
git clone https://github.com/melloddy/MELLODDY-TUNER.git
```
### Create enviroment 

Create your own enviroment from the given yml file with:

```
conda env create -f melloddy_pipeline_env.yml
```


The environment can be activated by:

```
conda activate melloddy_pipeline
```
This environment can be used for TUNER and SPARSECHEM.
## Package Installation

You have to install the melloddy-tuner package with pip:

```
pip install -e .
```
Make sure that the current version is installed.

# Input and Output Files for Year 3

## Preparation of Input Files (version year 3)

The following datasets can be generated with MELLODDY-TUNERfor the year 3 federated run:

1. **without** auxiliary data 

    *a)* **cls**: Classification data

    *b)* **reg**: Regression data

    *c)* **hybrid**: Classification & regression data


2. **with** auxiliary data (from HTS, images)

    *a)* **cls**: Classification data

    ~~*b)* **reg**: Regression data~~

    *c)* **hybrid**: Classification & regression data

Each pharma partner will needs to prepare:

1. Assay mapping/weight table **T0** prepared according to the data preparation manual.
2. Actvity data file **T1** linking information about activity via assay and compound identifiers.
3. Comprehensive structure file **T2** containing compound identifier and SMILES strings of all compounds present in T1.

*NEW in YEAR 3*: The script can handle multiple T0 and T1 files and concatenate these. Make sure that you do not have duplicated identifiers. 

Rules and guidelines to extract data from in-house databases can be found in the Data Preparation Manual provided by WP1.

To run the preprocessing script, the input files should be in csv format and should contain **all** following columns (even if they are empty):


### T0 weight table (T0) (Required column) 



| input_assay_id |assay_type|use_in_regression | is_binary | expert_threshold_1|expert_threshold_2|expert_threshold_3|expert_threshold_4|expert_threshold_5|direction|catalog_assay_id|parent_assay_id|
|-----------| ----------- | ----------- | ----------- |-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
|**needs to be unique**| **not empty** | **not empty** | **not empty** |*optional*|*optional*|*optional*|*optional*|*optional*|*optional*|*optional*|*optional*|



### T1 activity file 

| input_compound_id |input_assay_id|standard_qualifier | standard_value | 
|-----------| ----------- | ----------- | ----------- |
|**not empty** | **not empty**  | *defined values allowed* | **not empty**  |

### T2 structure file 

| input_compound_id |smiles|
|-----------| ----------- |
|**needs to be unique**| **not empty**  |


An example configuration and key file for standardization is provided in:

```
/config/example_parameters.json
/config/example_key.json
```

The configuration parameters used for standardization, fingerprints, activity data filtering must be set in a parameters.json file (see details [here](#parameter-definitions)). <br>
The given high entropy bits for LSH folding are derived from the ChEMBL25 compounds with fingerprint settings:  
 ```
 size: 32000
 radius: 3
 hashed: True  
 binarized: True
 ```

It is possible to shuffle the bits of the fingerprints with an encryption key examplified by a trivial encryption key provided in example_key.json. This is not required in non privacy preserving and federated scenarios (--key and --ref_hash not needed).<br>
The compound fold assignments is also dependent on the encryption key. <br>
In case an encryption key is used, MELLODDY TUNER can perform double checks to ensure the validity of the used encryption configuration and make sure of consistency of the data preparation across the involved parties. <br>
The "reference dataset" provided under `unit_test/refeence_files/reference_set.csv` is prepared using the input ecryption key. The prepared reference set is then hashed into a hash key which is compared to a reference hash key circulated across the parties. <br>



## Expected Output Files

Partners should run the pipeline with two datasets: (1) one **without** auxiliary data (`using_auxiliary == no`), (2) and one **with** auxiliary data (`using_auxiliary == yes`).\
You should have two defined output directories and the `matrices` folder containing subfolder(s):

| MELLODDY TUNER run | matrices subfolder | filename |  cls | reg |
| ----------- | ----------- |----------- |-----------|-----------|
| **wo_aux**|**cls** | cls_T11_x.npz|X||
|  || cls_T11_x_fold_vector.npy|X||
|  || cls_T10_y.npz |X||
|  || cls_weights.csv|X||
| |**reg**| reg_T11_x.npz||X|
|  || reg_T11_x_fold_vector.npy||X|
|  || reg_T10_y.npy||X|
|  || reg_T10_censor_y.npy||X|
|  || reg_weights.csv||X|
| |**hyb**| hyb_T11_x.npz|X||
|  || hyb_T11_x_fold_vector.npy|X||
|  || hyb_cls_T10_y.npz |X||
|  || hyb_cls_weights.csv|X||
|  || hyb_reg_T10_y.npy||X|
|  || hyb_reg_T10_censor_y.npy||X|
|  || hyb_reg_weights.csv||X|
| **w_aux** |**clsaux** | clsaux_T11_x.npz|X||
|  || clsaux_T11_x_fold_vector.npy|X||
|  || clsaux_T10_y.npz |X||
|  || clsaux_weights.csv|X||




# 1. Run Data Prepration Script


All steps can be executed with the commandline interface tool `tunercli`

The script allows the following commands:

```
tunercli
{
standardize_smiles      # standardization of SMILES
calculate_descriptors  # calculate fingerprints
assign_fold         #  assign folds
assign_lsh_fold     # assign LSH-based folds
agg_activity_data   # aggregate values
apply_thresholding  # apply thresholding to classification data
filter_classification_data  # filter classification tasks
filter_regression_data  # filter regression tasks
make_matrices           # create sparse matrices from dataframes
make_folders_s3       # creates folder structure ready to upload to S3
prepare_4_training     # Run the full pipeline to process data for training
prepare_4_prediction    # Run the full pipeline to process data for prediction
prepare_structure_data # Run the structure preparation pipeline (only structure related steps)
prepare_activity_data # Run the activity data preparation pipeline (after prepare_structure_data, only activity data related steps)
} 

```

**NEW in Year 3**: You execute all subcommands with a given `run_parameters.json` file instead of defining everything as arguments. The script will also automatically generate these json files when running the scripts with flags.
For example:

```
tunercli prepare_4_training --run_parameters config/run_parameters/pipeline.json
```
Multiple run_parameter json files can be found in `config/run_parameters/`.

Each execution results in a **run_report** which is automatically generated in
`output_dir/run_name/<DATE>_subcommand_run_report.json`. This report contains the run parameters, statistics about the preprocessing steps and information about passed/failed sanity checks.


All subcommands can be executed individually or in pipelines suited for training(`prepare_4_training`) or prediction (`prepare_4_prediction`) processing.

###  `prepare_4_training`
To standardize and prepare your input data and create ML-ready files, run the following command with arguments:\

**1.** path to your T2 structure file (--structure_file)\
**2.** path to your T1 activity file (--activity_file)\
**3.** path to your weight table T0 (--weight_table)\
**4.** path to the config file (--conf_file)\
**5.** path to the key file (--key_file)\
**6.** path of the output directory, where all output files will be stored (--output_dir)\
**7.** user-defined name of your current run (--run_name)\
**8.** tag `using_auxiliary` to identify dataset without (`no`) or with auxiliary data (`yes`)\
**9.** Folding method to assign test/validation/test splits. Choices: **`scaffold`** (**must be used in year 2!**) or `lsh` (year 1)\
**10.** (Optional) Number of CPUs to use during the execution of the script (default: 1) (--number_cpu)\
**11.** (Optional) JSON file with a reference hash key to ensure usage of the same paramters between different users. (--ref_hash)\
**12.** (Optional) Non-interactive mode for cluster/server runs.(--non_interactive) \



As an example, you can prepare your data for training by executing tunercli prepare_4_training:
```
tunercli prepare_4_training 
--structure_file {path/to/your/structure_file_T2.csv}
--activity_file {/path/to/your/activity_data_file_T1.csv}
--weight_table {/path/to/your/weight_table_T0.csv}
--config_file {/path/to/the/distributed/parameters.json}
--key_file {/path/to/the/distributed/key.json}
--output_dir {path/to/the/output_directory}
--run_name {name of your current run}
--using_auxiliary {no or yes}
--folding_method {scaffold or lsh}
--number_cpu {number of CPUs to use}
--ref_hash {path/to/the/provided/ref_hash.json}

```

In the given output directory the script will create a folder with the name of the "run_name" and three subfolders:
```
path/to/the/output_directory/run_name/results_tmp       # contain intermediate results from standardization, descriptors and activity data formatting
path/to/the/output_directory/run_name/results           # contain the final dataframe files with continuous IDs (T10c_cont, T10r_cont, T6_cont).
path/to/the/output_directory/run_name/mapping_table     # contain relevant mapping tables
path/to/the/output_directory/run_name/reference_set     # contain files for constistency check
path/to/the/output_directory/run_name/wo_aux or w_aux/matrices          # contain sparse matrices and meta data files for SparseChem
```

## `prepare_structure_data`

For processing only the structure data (first step), you can run:

```
tunercli prepare_structure_data 
--structure_file {path/to/your/structure_file_T2.csv}
--activity_file {/path/to/your/activity_data_file_T1.csv}
--weight_table {/path/to/your/weight_table_T0.csv}
--config_file {/path/to/the/distributed/parameters.json}
--key_file {/path/to/the/distributed/key.json}
--output_dir {path/to/the/output_directory}
--run_name {name of your current run}
--using_auxiliary {no or yes}
--folding_method {scaffold or lsh}
--number_cpu {number of CPUs to use}
--ref_hash {path/to/the/provided/ref_hash.json}

```

This will process the input structures only and is required before you prepare your activity data.

## `prepare_activity_data`

For processing only the activity data (second step), you can run:

```
tunercli prepare_activity_data 
--mapping_table path/to/your/mapping_table/T5.csv
--T6_file path/to/your/mapping_table/T6.csv
--activity_files path/to/your/T1.csv
--weight_tables path/to/your/T0.csv
--catalog_file path/to/reference-file/T_cat.csv
--config_file {/path/to/the/distributed/parameters.json}
--key_file {/path/to/the/distributed/key.json}
--output_dir {path/to/the/output_directory}
--run_name {name of your current run}
--using_auxiliary {no or yes}
--number_cpu {number of CPUs to use}
--ref_hash {path/to/the/provided/ref_hash.json}
}
```
After executing both steps sequentially, you processed all your data ready for SparseChem.

## `prepare_4_prediction` 

For predicting new compounds with an already trained ML model, only a structure file (like T2.csv) has to be preprocessed.

To standardize and prepare your input data for prediction, run the following command with arguments:\
**1.** path to your T2 structure file (--structure_file)\
**2.** path to the config file (--config_file)\
**3.** path to the key file (--key_file)\
**4.** path of the output directory, where all output files will be stored (--output_dir)\
**5.** user-defined name of your current run (--run_name)\
**6.** (Optional) Number of CPUs to use during the execution of the script (default: 2) (--number_cpu)\
**7.** (Optional) JSON file with a reference hash key to ensure usage of the same parameters between different users. (--ref_hash)\
**8.** (Optional) Non-interactive mode for cluster/server runs. (--non_interactive)

For example, you can run:
```
tunercli prepare_4_prediction \

--structure_file {path/to/your/structure_file_T2.csv}\
--config_file {/path/to/the/distributed/parameters.json}\
--key_file {/path/to/the/distributed/key.json}\
--output_dir {path/to/the/output_directory}\
--run_name {name of your current run}\
--number_cpu {number of CPUs to use}\
--ref_hash {path/to/the/provided/ref_hash.json}\
```

In the given output directory the script will create a folder with the name of the "run_name" and three subfolders:
```
path/to/the/output_directory/run_name/results_tmp       # contain intermediate results from standardization, descriptors and activity data formatting
path/to/the/output_directory/run_name/results           # contain the final dataframe files with continuous IDs (T10c_cont, T10r_cont, T6_cont).
path/to/the/output_directory/run_name/mapping_table     # contain relevant mapping tables
path/to/the/output_directory/run_name/reference_set     # contain files for constistency check
path/to/the/output_directory/run_name/matrices          # contain sparse matrices and meta data files for SparseChem
```



# 2 Individual scripts

The data processing includes several steps, which can be performed independently from each other.<br>
For the following examples, these file paths need to be defined :<br>

## Input file paths definition

```
# configuration parameters (needs adjustement at your setup)
param=<path to data_prep/config/example_parameters.json>
key=<path to data_prep/config/example_key.json>
ref=<path to data_prep/unit_test/reference_files/ref_hash.json> 
outdir=<path to output folder>
run_name=<data prep run name>
num_cpu=<number of cpus for multi-threaded processes>

# melloddy tuner initial input files (needs adjustement at your setup)
t0=<path to initial T0.csv (assays)>
t1=<path to initial T1.csv (activities)>
t2=<path to initial T2.csv (smiles)>

# melloddy tuner intermediate files (these definition are static)
t2_std=$outdir/$run_name/results_tmp/standardization/T2_standardized.csv # standardized structures (output from standardize_smiles)
t2_desc=$outdir/$run_name/results_tmp/descriptors/T2_descriptors.csv     # descriptors of structures (output of calculate_descriptors)
t5=$outdir/$run_name/mapping_table/T5.csv                                # mapping table (input_compound_id->descriptor_vector_id->fold_id)
t4c=$outdir/$run_name/results_tmp/thresholding/T4c.csv                   # classification tasks activity labels (output of apply_thresholding)
t3c=$outdir/$run_name/results_tmp/thresholding/T3c.csv                   # classification tasks annotations (output of apply_thresholding)
t4r=$outdir/$run_name/results_tmp/aggregation/T4r.csv                    # regression tasks activity data (output of agg_activity_data)
t6=$outdir/$run_name/mapping_table/T6.csv                                # mapping table (descriptor_vector_id->fp_feat->fp_val->fold_id)
t8c=$outdir/$run_name/results_tmp/classification/T8c.csv                 # classification tasks annotations (includes continuous identifiers, class counts and perf aggr flags)
t8r=$outdir/$run_name/results_tmp/regression/T8r.csv                     # regression tasks annotations (includes continuous identifiers)
t10c=$outdir/$run_name/results_tmp/classification/T10c.csv               # classification tasks (continous task identifiers, descriptor vectors, fold assignment, class label)
t10r=$outdir/$run_name/results_tmp/regression/T10r.csv                   # regression tasks (continous task identifiers, descriptor vectors, fold assignment, activity, qualifier)
```

## 2.1  `standardize_smiles`

Script `standardize_smiles` takes the input smiles csv file and standardizes the smiles according to pre-defined rules.<br>
Please refer to section "Input file paths definition" for details on the input files.

For example, you can run:

```
tunercli standardize_smiles  --structure_file $t2 \
                             --config_file $param \
                             --key_file $key \
                             --output_dir $outdir \
                             --run_name $run_name \
                             --number_cpu $num_cpu \
                             --non_interactive 
#                             --ref_hash $ref
```

Produces : 
```
output_dir/
└── run_name
    └── results_tmp
        └── standardization
            ├── T2_standardized.csv
            └── T2_standardized.FAILED.csv

```



## 2.2 `calculate_descriptors`

Script `calculate_descriptors` calculates a descriptor based on the standardized smiles, and scramble features with given key. Use an input file containing standardized SMILES (`canonical_smiles`) and a fold id (`fold_id`).
Please refer to section "Input file paths definition" for details on the input files.


For example, you can run:

```
tunercli calculate_descriptors --structure_file $t2_std \
                               --config_file $param \
                               --key_file $key \
                               --output_dir $outdir \
                               --run_name $run_name \
                               --number_cpu $num_cpu \
                               --non_interactive
#                               --ref_hash $ref


```

Produces:
```
output_dir/
└── run_name
    └── results_tmp
        └── descriptors
            ├── T2_descriptors.csv
            └── T2_descriptors.FAILED.csv

```


## 2.3.1 `assign_fold` with the scaffold-based split 

Script `assgin_fold` assign fold identifiers by a scaffold-based approach using a input file with standardized SMILES (`canonical_smiles` as column name), and the descriptors (`fp_feat` and `fp_val`), i.e. `results_tmp/descriptors/T2_descriptors.csv`.<br>
Please refer to section "Input file paths definition" for details on the input files.

For example, you can run:

```
tunercli assign_fold --structure_file $t2_desc \
                     --config_file $param \
                     --key_file $key \
                     --output_dir $outdir \
                     --run_name $run_name \
                     --number_cpu $num_cpu \
                     --non_interactive \
#                     --ref_hash $ref


```

Produces: 
```
output_dir
└── run_name
    ├── mapping_table
    │   ├── T5.csv
    │   └── T6.csv
    └── results_tmp
        └── folding
            ├── T2_descriptor_vector_id.DUPLICATES.csv
            ├── T2_folds.csv
            └── T2_folds.FAILED.csv

```



## 2.3.2 `assign_lsh_fold` with Locality Sensitive Hashing (LSH) 

Script `assgin_fold` assign fold identifiers by <b>L</b>ocality <b>S</b>ensitive <b>H</b>ashing approach using a input file with standardized SMILES (`canonical_smiles` as column name), and the descriptors (`fp_feat` and `fp_val`), i.e. `results_tmp/descriptors/T2_descriptors.csv`.<br>
Please refer to section "Input file paths definition" for details on the input files.

For example, you can run:

```
tunercli assign_lsh_fold --structure_file $t2_desc \
                         --config_file $param \
                         --key_file $key \
                         --output_dir $outdir \
                         --run_name $run_name \
                         --number_cpu $num_cpu \
                         --non_interactive \
#                         --ref_hash $ref


```




## 2.4 `agg_activity_data`
Script `aggregate_values.py` removes activity data that is outside of the credibility range as provided in the parameter file, standardizes qualifiers to {<,>,=} and aggregates replicates that appeared due to structure standardization. It creates table T4r and some additional files for logging data that is outside the credibility range or couldn't be aggregated based on T0, T1 and T5. 
Please refer to section "Input file paths definition" for details on the input files.

```
tunercli agg_activity_data --assay_file $t0 \
                           --activity_file $t1 \
                           --mapping_table $t5 \
                           --config_file $param \
                           --key_file $key \
                           --output_dir $outdir \
                           --run_name $run_name \
                           --number_cpu $num_cpu \
                           --non_interactive 
#                           --ref_has $ref \
#                           --reference_set $ref_set
```

Produces:
```
output_dir/run_name/results_tmp/aggregation/
├── aggregation.log
├── duplicates_T1.csv
├── failed_aggr_T1.csv
├── failed_range_T1.csv
├── failed_std_T1.csv
└── T4r.csv

```



## 2.5 `apply_thresholding`
Script `apply_thresholding` sets thresholdds for classification tasks without given expert thresholds.
Please refer to section "Input file paths definition" for details on the input files.

For example, you can run:
```
tunercli apply_thresholding --activity_file $t4r \
                            --assay_file $t0 \
                            --config_file $param \
                            --key_file $key \
                            --output_dir $outdir \
                            --run_name $run_name \
                            --number_cpu $num_cpu \
                            --non_interactive 
#                            --ref_has $ref \
#                            --reference_set $ref_set
```

Produces: 
```
output_dir/run_name/results_tmp/thresholding/
├── T3c.csv
├── T4c.csv
└── T4c.FAILED.csv

```



## 2.6 classification tasks filtering
Script `filter_classification` filters out classification assays based on the provided quorum and set weights for training. It produces the tables T10c and T8c. 
Please refer to section "Input file paths definition" for details on the input files.

```
tunercli filter_classification_data --classification_activity_file $t4c \
                                    --classification_weight_table $t3c \
                                    --config_file $param \
                                    --key_file $key \
                                    --output_dir $outdir \
                                    --run_name $run_name \
                                    --non_interactive
#                                    --ref_hash $ref 
#                                    --reference_set $ref_set
```

Produces: 
```
output_dir/run_name/results_tmp/classification/
├── duplicates_T4c.csv
├── filtered_out_T4c.csv
├── T10c.csv
└── T8c.csv

```



## 2.7 regression tasks filtering

Script `filter_regression` filters out regression assays based on the provided quorum and set weights for training. It produces the tables T10r and T8r. 
Please refer to section "Input file paths definition" for details on the input files.

```
tunercli filter_regression_data --regression_activity_file $t4r \
                                --regression_weight_table $t0 \
                                --config_file $param \
                                --key_file $key \
                                --output_dir $outdir \
                                --run_name $run_name \
                                --non_interactive 
#                                --ref_hash $ref
#                                --reference_set $ref_set
```

Produces: 
```
output_dir/run_name/results_tmp/regression/
├── duplicates_T4r.csv
├── filtered_out_T4r.csv
├── T10r.csv
└── T8r.csv

```


## 2.8 `make_matrices`
Script `make_matrices` formats the result csv files into ML ready matrix formats.
Please refer to section "Input file paths definition" for details on the input files.

For example, you can run:

```
tunercli make_matrices  --structure_file $t6 \
                        --activity_file_clf $t10c \
                        --weight_table_clf $t8c \
                        --activity_file_reg $t10r \
                        --weight_table_reg $t8r \
                        --config_file $param \
                        --key_file $key \
                        --output_dir $outdir \
                        --run_name $run_name \
                        --using_auxiliary {no or yes} \
                        --non_interactive
#                        --ref_hash $ref 
```

`--using_auxiliary no` is suitable to use when the tuner input files do not contain auxiliary data. The `--using_auxiliary no` will ensure the creation of the cls and reg subdirectories: 

```
output_dir/run_name/matrices/
├──wo_aux
    ├── cls/
    │     ├── cls_T10_y.npz
    │     ├── cls_T11_fold_vector.npy
    │     ├── cls_T11_x.npz
    │     └── cls_weights.csv    
    ├── reg/
    │     ├── reg_T10_censor_y.npz
    │     ├── reg_T10_y.npz
    │     ├── reg_T11_fold_vector.npy
    │     ├── reg_T11_x.npz
    │     └── reg_weights.csv    
    └── hyb/
          ├── hyb_cls_T10_y.npz
          ├── hyb_T11_x.npz
          ├── hyb_T11_fold_vector.npy
          ├── hyb_cls_weights.csv
          ├── hyb_reg_T10_censor_y.npz
          ├── hyb_reg_T10_y.npz
          └── hyb_reg_weights.csv

output_dir/run_name/results
├── T10c_cont.csv
├── T10r_cont.csv
└── T6_cont.csv

```

Or produces for data with aux. data (`--using_auxiliary yes`): 
```
output_dir/run_name/matrices/
├──w_aux
    ├── clsaux/
       ├── clsaux_T10_y.npz
       ├── clsaux_T11_fold_vector.npy
       ├── clsaux_T11_x.npz
       └── clsaux_weights.csv    
    


output_dir/run_name/results
├── T10c_cont.csv
├── T10r_cont.csv
└── T6_cont.csv
```


## 2.9 `make_folders_s3`
Script `make_folders_s3` creates the required folders for S3 bucket.


For example, you can run:

```
tunercli make_folders_s3  --config_file $param \
                        --key_file $key \
                        --output_dir $outdir \
                        --run_name $run_name \
                        --using_auxiliary {no or yes} \
                        --non_interactive
#                        --ref_hash $ref 
```

`--using_auxiliary no` is suitable to use when the tuner input files do not contain auxiliary data (`cls`, `reg` and `hyb` are considered). Or produces for data with auxiliary data (`--using_auxiliary yes`) to create `clsaux` subfolders.



# Parameter definitions

This section describes the parameters to be used to prepare a dataset with MELLODDY TUNER examplified in `config/example_parameters.json`: <br>


### standardization 
- <b>max_num_tautomers</b>: maximum number of enumerated tautomers.
- <b>max_num_atoms</b>: maximum number of (heavy) atoms allowed.
- <b>include_stereoinfo</b>: (true or fasle) defines if stereochemistry shoudl be considered during the standardization process.

### fingerprint
- <b>radius</b>: Morgan fingerprint radius 
- <b>hashed</b>: true or false leads to use of rdkit `GetHashedMorganFingerprint` or `GetMorganFingerprint` respectively 
- <b>fold_size</b>: number of bits in the fingerprint
- <b>binarized</b>: (true or false), false indicating the fingerprint should be counts rather than binary bits (not supported yet)

### scaffold_folding
- <b>nfolds</b>: if scaffold based dataset split in use, defines the number of folds the dataset will be split in.

### credibility_range
Defines the standard_value (activity data points) credibility ranges per input_assay_id, values falling outside will be discarded. Ranges are defined per assay_type category.<br>
- <b>min</b>: minimum allowed value
- <b>max</b>: maximum allowed value
- <b>std</b>: minimum allowed standard deviation allowed across the (uncensored) standard_values of an input_assay_id. An assay with lower stdand deviation will be discarded.

### train_quorum
Defines the minimum amount of data that is required for a task to take place in the prepared dataset, hence to participate in the model training. <br>
Quorums are defined per assay_type category and per modelling type (regression or classification). <br>

#### regression
- <b>num_total</b>: minumum number of data points an assay requires to become a task in the prepared dataset (including censored data points, <i>i.e.</i> data points associated to `<`, `>` standard_relation)
- <b>num_uncensored_total</b>: minimum number of data points an assay requires to become a task in the prepared dataset (data points associated to `=` standard_relation only)

#### classification
- <b>num_active_total</b>: minimum number of positive samples a classifiation task requires to make it to the prepared dataset
- <b>num_inactive_total</b>: minimum number of negative samples a classification task requires to make it to the prepared dataset

### evaluation_quorum
Defines the minimum amount of data that is required for a task to be given an aggregation_weight=1 (see sparsechem task weights), hence to contribute to the aggregate performance calculation. <br>
Quorums are defined per assay_type category and per modelling type (regression or classification) and must be valid in each of the fold splits. <br>

#### regression
- <b>num_fold_min</b>: minimum number of data points across all the fold splits (ensures each fold split has at least this amound of data points,  data points associated to `=`, `<` or `>` standard_relation)
- <b>num_uncensored_fold_min</b> : minimum number of uncensored data points (data points associated to `=` standard_relation only)

#### classification
- <b>num_active_fold_min</b>: minimum number of positive samples across all the fold splits
- <b>num_inactive_fold_min</b>: minimuym number of negatives samples across all the fold splits

### initial_task_weights
Defines the initial tasks training weights (see sparsechem task weights) per assay_type category

### global_thresholds
Defines global thresholds for assay_types to be applied to devise classification tasks (other tasks have either user defined expert_thresholds set in T0 or will be attributed a threshold automatically, see step 2.5 `apply_thresholding`). 

### censored_downweighting
- <b>knock_in_barrier</b>: in regression models censored data points (<i>i.e. data points associated to `<` or `>` standard_relation </i> ) can be downweighted to reduce their contributions to the loss function of sparsechem (see sparsechem regression task weights). The downweighting kicks-in if the fraction of censored data of a task is higher than the <b>knock_in_barrier</b>

### count_task
- <b>count_data_points</b>: <i>Year 1 parameter</i>

### lsh
- <b>nfolds</b>: if LSH data split in use, defines the number of folds the dataset will be split in. 
- <b>bits</b>: list of high entropy bits keys to be used for fold assignment by the LSH split methodology. This list is specific to a fingerprint type. <i>e.g.</i> the list provided in the example_parameter.json file was built from a public data set of compounds as described by their Morgan binary hashed (length 32k) fingerprints. Not suitable if LSH to be used with a different fingerprint definition than that in the config/example_pararameters.json file.
