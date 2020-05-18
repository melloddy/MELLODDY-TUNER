# Data Preparation Script for the IMI Project MELLODDY
Data preparation scripts, including locality sensitive hashing (LSH).
## Requirements
The data preprocessing script requires:
1. Python 3.6 or higher
2. Local Conda installation (e.g. miniconda)
3. Git installation


## Setup the environment

First, clone the git repository from the MELLODDY gitlab repository:
```
git clone git@git.infra.melloddy.eu:wp1/data_prep.git
```

Then, you can install the conda environment "melloddy_data_prep" and the required packages by runnning the following command:
```
sh install_environment.sh
```
A list with all installed packages can be found in the file: environment_melloddy_data_prep.yml

The environment can be activated by:
```
conda activate melloddy_tuner
```

After the activationof the environment, you have to install the melloddy-tuner package with pip:
```
pip install -e .
```

Finally, you can run the processing scripts located in ```git_repo/bin/```.

## Prepare Input Files
The input files contain information about structures (T2) and activity data of these structures (T4) in certain assays.\
A weight table file describes the input assays and their corresponding weights in multi-class prediction setup.\
Rules and guidelines to extract data from in-house databases can be found in the Data Preparation Manual provided by WP1.

To run the preprocessing script, the input files should be in csv format and should contain:

**a)** structure file (T2) containing 2 columns with the headers:\
**1.** input_compound_id\
**2.** smiles

**b)** activity file (T4) containing 3 columns with the headers:\
**1.** input_compound_id\
**2.** classification_task_id\
**3.** class_label

**c)** weight table (T3) containing the following columns with headers:\
**1.** classification_task_id\
**2.** input_assay_id  (*optional*, your own assay identifier)\
**3.** assay_type\
**4.** weight (individual weight of each assay for multi-class predictions)

Additional columns are allowed in this weight table.

An example configuration file for standardization is provided in:
```
/tests/structure_preparation_test/example_parameters.json
```
containing information about structure standardization options, fingerprint settings,\
encryption key, high entropy bits for train/test splitting with LSH and activity data thresholds.
The config file can also be imported by the user.


## Run Data Prepration Script for Training

To standardize and prepare your input data and create ML-ready files, run the following command with arguments:\


**1.** path to your T2 structure file (--structure_file)\
**2.** path to your T4 activity file (--activity_file)\
**3.** path to your weight table T3 (--weight_table)\
**4.** path to the config file (--conf_file)\
**5.** path of the output directory, where all output files will be stored (--output_dir)\
**6.** user-defined name of your current run (--run_name)\
**7.** (Optional) Number of CPUs to use during the execution of the script (default: 2) (--number_cpu)\
**8.** (Optional) JSON file with a reference hash key to ensure usage of the same paramters between different users. (--ref_hash)\
**9.** (Optional) Non-interactive mode for cluster/server runs.(--non_interactive) \
As an example, you can prepare your data for training by executing prepare_4_melloddy.py:
```
python bin/prepare_4_melloddy.py \
--structure_file {path/to/your/structure_file_T2.csv}\
--activity_file {/path/to/your/activity_data_file_T4.csv}\
--weight_table {/path/to/your/weight_table_T3.csv}\
--config_file {/path/to/the/distributed/parameters.json}\
--output_dir {path/to/the/output_directory}\
--run_name {name of your current run}\
--number_cpu {number of CPUs to use}\
--ref_hash {path/to/the/provided/ref_hash.json}\
```

In the given output directory the script will create a folder with the name of the "run_name" and three subfolders:
```
path/to/the/output_directory/run_name/files_4_ml
path/to/the/output_directory/run_name/results
path/to/the/output_directory/run_name/results_tmp
```

The folder "results" contains the files which the model will use for the predictions (T11 and T10, and a T10 aggregated by counts, and the weight tables T3_mapped and T9).
The folder "results_tmp" contains subfolders for standardization, descriptors and activity formatting including mapping tables and additional output files to track duplicates, failed entries or excluded data.
The folder "files_4_ml" contains files which are ready to run the machine learning scripts.
The script generates two mtx files (for structure (X), and activity (Y) data) and the folding vector as npy file. It also contains a copy of T10_counts.csv and a reduced version of the weight table T9 (weight_table_T9_red.csv)

The script will also generate a json file ("generated_hash.json") containing a hash key based on a reference set to ensure that every partner uses the same parameters.
If a "ref_hash.json" is provided by the user, the "generated_hash.json" will be compared to it and the will stop, if the keys do not match.

An example reference hash key file for ```example_parameters.json```  is given in:
```
/tests/structure_preparation_test/ref_hash.json
```


## Run Data Preparation Script for Prediction

For predicting new compounds with an already trained ML model, only a structure file (like T2.csv) has to be preprocessed.
To prepare your structure files, please add the argument ```--prediction_only``` when running the two scripts.

To standardize and prepare your input data for prediction, run the following command with arguments:\
**1.** Add the argument ```--prediction_only``` to run process only structure data\
**2.** path to your T2 structure file (--structure_file)\
**3.** path to the config file (--config_file)\
**4.** path of the output directory, where all output files will be stored (--output_dir)\
**5.** user-defined name of your current run (--run_name)\
**6.** (Optional) Number of CPUs to use during the execution of the script (default: 2) (--number_cpu)\
**7.** (Optional) JSON file with a reference hash key to ensure usage of the same paramters between different users. (--ref_hash)\
**8.** (Optional) Non-interactive mode for cluster/server runs. (--non_interactive)

For example, you can run:
```
python bin/prepare_4_melloddy.py \
--prediction_only\
--structure_file {path/to/your/structure_file_T2.csv}\
--config_file {/path/to/the/distributed/parameters.json}\
--output_dir {path/to/the/output_directory}\
--run_name {name of your current run}\
--number_cpu {number of CPUs to use}\
--ref_hash {path/to/the/provided/ref_hash.json}\
```





## Run individual scripts

The data processing includes 3 different steps, which can be performed independently from each other.

1. ```bin/standardize_smiles.py``` takes the input smiles csv file and standardizes the smiles according to pre-defined rules.
2. ```bin/calculate_descriptors.py``` calculates a descriptor based on the standardized smiles, hash the descriptors with the given key and split the data into a given number of folds using a locality-sensitive hashing.
3. ```bin/activity_data_formatting.py``` formats the input bioactivity data into the required output format considering pre-defined rules.
4. ```bin/hash_reference_set.py``` standardize a reference set of molecules as a unit test to ensure that the same configuration was used. 
5. ```bin/csv_2_mtx.py``` formats the result csv files into ML ready data formats.


## Comparison with Reference Result Files


To verify the common script, please run the pipeline with the provided public data sets (chembl_T2.csv, chembl_T3.csv,  chembl_T4.csv).  
[ChEMBL input files on BOX (public)](https://jjcloud.box.com/s/ks44jfvex6hq5ycm9etyyronmh77qot0)

Please use the config file:
```
tests/structure_preparation_test/example_parameters.json
```
And as reference hash file:
```
tests/structure_preparation_test/ref_hash.json
```

To process the given ChEMBL files, run the following code:

```
python bin/prepare_4_melloddy.py \
--structure_file {path/to//chembl_T2.csv}\
--activity_file {/path/to/chembl_T4.csv}\
--weight_table {/path/to/chembl_T3.csv}\
--config_file {/path/to/tests/structure_preparation_test/example_parameters.json}\
--output_dir {path/to/the/output_directory}\
--run_name {name of your current run}\
--number_cpu {number of CPUs to use}\
--ref_hash {path/to/tests/structure_preparation_test/ref_hash.json}\
```

The public ref_hash.json file and the related processed files for comparison are on BOX (8th of May 2020, Version 1.0):  
[ChEMBL process files on BOX (public)](https://jjcloud.box.com/s/tj45v0584p3zexq1ma83ok7it1rn7oe5)

Please compare in particular the following files (sorted by descriptor_vector_id):

**a)** results/T11.csv

**b)** results/T10.csv

**c)** results/T10_counts.csv

**d)** results/weight_table_T3_mapped.csv

**e)** results/weight_table_T9.csv

**f)** results_tmp/descriptors/mapping_table_T5.csv

**g)** results_tmp/descriptors/mapping_table_T10.csv

The files_4_ml folder consists of 5 files, which can be compared to the reference files:

**a)** ```files_4_ml/T11_x.mtx``` and ```files_4_ml/T11_x.npy```

**b)** ```files_4_ml/T11_fold_vector.npy```

**c)** ```files_4_ml/T10_counts.csv```

**d)** ```files_4_ml/T10_y.mtx``` and ```files_4_ml/T10_y.npy```

**e)** ```files_4_ml/T9_red.csv```


Pharma ONLY:
You find the private "ref_hash.json" and the related processed files in the PharmaOnly BOX (8th of May 2020, Version 1.0):  
[ChEMBL process files on BOX (private)](https://jjcloud.box.com/s/ocqkj6e3div2rcmi88f8fdw5jzkvyubu)


# Docker

## Build the docker image

In order to build the docker image on your computer simply run:
```
docker build -t melloddy/data_prep .
```


## Run the Data Preparation using the docker image

This should not be officially used right now, but is made available in case it is of use for testing notably here:

### Prerequisit

You need to build the docker image prior to running it on your machine

### Command Line

```docker run -v $PWD/examples:/data -v $PWD/parameters.json:/params/parameters.json -w /opt/data_prep melloddy/data_prep conda run -n melloddy-tuner python bin/prepare_4_melloddy.py --structure_file {/data/path/to/your/structure_file_T2.csv}\
--activity_file {/data/path/to/your/activity_data_file_T4.csv}\
--weight_table {/data/path/to/your/weight_table_T3.csv}\
--config_file /params/parameters.json\
--output_dir {/data/path/to/the/output_directory}\
--run_name {name of your current run}\
--number_cpu {number of CPUs to use}
--ref_hash {path/to/the/provided/ref_hash.json}
```


Example command line on Chembl reference test-set:
```
docker run -v $PWD/chembl:/chembl  melloddy/data_prep conda run -n melloddy_tuner python bin/prepare_4_melloddy.py --structure_file /chembl/chembl_T2.csv --activity_file /chembl/chembl_T4.csv --weight_table /chembl/chembl_T3.csv --config_file /opt/data_prep/melloddy_tuner/parameters.json --output_dir /chembl --number_cpu 4 --run_name chembl_test --ref_hash /chembl/ref_hash.json
```




<!-- ## Comparison of Prediction results
Furthermore, the provided model (chembl_test) was trained with the public data set (chembl_T2_T11_csr_matrix_x.mtx, chembl_T4_T10_csr_matrix_y.mtx) with the following hyperparameters (check WP2 readme for more details):
```
python train.py   
--x                 chembl_T2_T11_csr_matrix_x.mtx  
--y                 chembl_T4_T10_csr_matrix_y.mtx  
--folding           chembl_T2_T11_fold_vector.npy   
--fold_va           0   
--batch_ratio       0.02   
--hidden_sizes      400   
--last_dropout      0.2   
--middle_dropout    0.2   
--weight_decay      0.0   
--epochs            20   
--lr                1e-3   
--lr_steps          10  
--lr_alpha          0.3 
--filename          chembl_test
```

Please perform a prediction with the reference data set (tests/structure_preparation_test/reference_set.csv) and this the provided model:
1. Perform standardization and formatting in prediction mode of the reference set:
```
python bin/prepare_4_melloddy.py 
-s tests/structure_preparation_test/reference_set.csv 
-c parameters.decrypted.json 
-o  /output_reference_set/
-n 256 
--prediction_only
```
```
python bin/csv_2_mtx.py 
-s /output_reference_set/results/reference_set_T11.csv 
-o /output_reference_set/
-g /output_reference_set/results/generated_hash.json 
-r /path_to_ref_hash.json
-cf /parameters.decrypted.json 
-m /output_chembl_training/files_4_ml_mapping_tables/chembl_T2_T11_map_bits2columns_X.csv
--prediction_only 
```
2. Start prediction of the reference set with the provided model (2020-02-05_chembl_models_cpu):
```
python predict.py     
--x /output_reference_set/files_4_ml_pred_only/reference_set_T11_csr_matrix_x.mtx     
--outfile /y_hat_reference_set.npy    
--conf models/2020-02-05_chembl_test-conf_cpu.npy     
--model models/2020-02-05_chembl_test_cpu.pt 
--dev cpu
```

3. Please compare the y_hat_reference_set.npy with the given example on box. 
You can do this by loading both npy arrays and compare the values within a certain threshold:
```
import numpy as np
y_hat = np.load('path_to_your_y_hat_npy_file')
y_hat_ref = np.load('path_to_the_provided_y_hat_npy_file')
np.allclose(y_hat, y_hat_ref, rtol=1e-5)
```
 -->