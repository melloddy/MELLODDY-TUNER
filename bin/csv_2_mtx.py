import argparse
import json
import os

from datetime import datetime

import shutil
import time
from itertools import chain
from pathlib import Path

import melloddy_tuner as tuner
import numpy as np
import pandas as pd
from scipy.io import mmwrite
from scipy.sparse import csr_matrix


def init_arg_parser():
    parser = argparse.ArgumentParser(description='Run formating script.')

    parser.add_argument('-c', '--config_file', type=str, help='path of the config file', required=True)

    parser.add_argument('-o', '--output_dir', type=str, help='path to output directory',
                        required=True)
    parser.add_argument('-r', '--run_name', type=str, help='name of your current run', required=True)
    parser.add_argument('-rh', '--ref_hash', type=str,
                        help='path to the reference hash key file provided by the consortium. (ref_hash.json)',
                        required=True)
    parser.add_argument('-p', '--prediction_only', 
                        help='Preprocess only chemical structures for prediction mode', action='store_true', default = False)
    parser.add_argument('-ni','--non_interactive', help='Enables an non-interactive mode for cluster/server usage', action='store_true', default=False)


    args = parser.parse_args()


    return args



def matrix_from_strucutres_for_prediction(df,bit_size):
    """
#     :param pandas df containing T11 (ecfps)
    """
    # create a dictionary mapping ecfp hash keys to column indices in X.mtx used for training 
    # ofc, the file here should be passed as a new argument 


    df['fp_json'] = df['fp_json'].str[1:-1]
    cols = df.columns.difference(['fp_json'])
    bits = df.fp_json.str.split(',')
    x_ijv_df = df.loc[df.index.repeat(bits.str.len()), cols].assign(bits=list(chain.from_iterable(bits.tolist())))
    x_ijv_df['bits'] = x_ijv_df['bits'].astype(np.int64)


    # works only for binary here: but could pick up the values in column fp_val
    x_ijv_df['value'] = np.ones(x_ijv_df.shape[0]).astype(np.int32)
    data = x_ijv_df['value'].values

    # get the row coordinates of the X matrix
    I, J = x_ijv_df['cont_descriptor_vector_id'], x_ijv_df['bits']

    # create the matrix, make sure it has the right dimension: here [number molecules to predict x number  defined by the parameter 'fold_size']
    matrix = csr_matrix((data.astype(float), (I, J)), shape=(df.shape[0], bit_size))

    return matrix


def matrix_from_structures(df, bit_size):
    df['fp_json'] = df['fp_json'].str[1:-1]
    cols = df.columns.difference(['fp_json'])
    bits = df.fp_json.str.split(',')
    x_ijv_df = df.loc[df.index.repeat(bits.str.len()), cols].assign(bits=list(chain.from_iterable(bits.tolist())))
    x_ijv_df['bits'] = x_ijv_df['bits'].astype(np.int64)

    ## Works for binary bits
    x_ijv_df['value'] = np.ones(x_ijv_df.shape[0]).astype(np.int32)
    data = x_ijv_df['value'].values

    I, J = x_ijv_df['cont_descriptor_vector_id'], x_ijv_df['bits']
    # create the scipy csr matrix
    matrix = csr_matrix((data.astype(float), (I, J)), shape=(df.shape[0], bit_size))
    return matrix


def matrix_from_activity(df):
    map_rows = {val: ind for ind, val in enumerate(np.unique(df['cont_descriptor_vector_id']))}
    map_cols = {val: ind for ind, val in enumerate(np.unique(df['cont_classification_task_id']))}
    
    data = df['class_label'].values

    I, J = [map_rows[x] for x in df['cont_descriptor_vector_id']], [map_cols[x] for x in
                                                                    df['cont_classification_task_id']]
    # create the scipy csr matrix
    matrix = csr_matrix((data.astype(float), (I, J)))
    return matrix


def folding_from_structure(df):
    folding_vector = df['fold_id'].values
    return folding_vector





def main(args, overwriting = True):
    """
    Main function reading input files, executing functions and writing output files.
    """
    ####################################
    # get parameters

    start = time.time()

    
    s_path = Path(args['output_dir'])
    s_path.mkdir(exist_ok=True)
    run_name = args['run_name']
    output_dir = s_path / run_name
    
    config_file = Path(args['config_file'])
    if config_file.is_file() is False:
            print('Config file does not exist.')
            quit()
   
     
    if args['ref_hash'] is None:
        print('No reference hash given. Comparison of generated and reference hash keys will be skipped.')
    else:
        with open(args['ref_hash']) as ref_hash_f:
            ref_hash = json.load(ref_hash_f)
        key_ref = ref_hash['unit_test_hash']
        path_gen_hash = output_dir / 'generated_hash.json'
        with open(path_gen_hash) as hash_f:
            key = json.load(hash_f)
        if key['unit_test_hash'] != key_ref:
                print('Different reference key. Please check the parameters you used for structure preparation.')
                quit()
    
    
    
    if args['prediction_only'] is True:
        print('Formatting data ready for predictions with a ML model.')

        T11_structure_file = output_dir / 'results' / 'T11_prediction_only.csv'
        path_files_4_ml = output_dir / 'files_4_ml_pred_only'
        path_files_4_ml.mkdir(exist_ok=True)
    else:
        print('Formatting data ready for training a ML model.')

        T11_structure_file = output_dir / 'results' / 'T11.csv'
        T10_activity_file = output_dir / 'results' / 'T10.csv'
        T10_activity_file_counts = output_dir / 'results' / 'T10_counts.csv'
        T9_weight_table_file = output_dir / 'results' / 'weight_table_T9.csv'
        if T10_activity_file.is_file() is False:
            print('Activity file does not exist.')
            quit()
        if T10_activity_file_counts.is_file() is False:
            print('Activity count file does not exist.')
            quit()
        if T9_weight_table_file.is_file() is False:
            print('Weight table file does not exist.')
            quit()
        path_files_4_ml = output_dir / 'files_4_ml'
        path_files_4_ml.mkdir(exist_ok=True)
        
        
    if T11_structure_file.is_file() is False:
                print('Structure file does not exist.')
                quit()        
       
    if overwriting is False:  
        if os.listdir(path_files_4_ml):
            override = input(f'Do you want to override files in {path_files_4_ml}?(type y or Y) \n The script will be aborted if you type anything else. ')
            if override == 'y' or override=='Y':
                 print(f'Files for run name {run_name} will be overwritten.')
            else:
                print('Processing aborted. Please change the run name and re-run the script.')
                quit()
    
    # Read config file and get fold size as matrix dimension
    fp_param = tuner.config.parameters.get_parameters(path=config_file)['fingerprint']
    bit_size = fp_param['fold_size']
    
    # Preparing structure-related X matrix
    T11_structure_df = tuner.read_csv(T11_structure_file)
    structure_matrix = matrix_from_strucutres_for_prediction(T11_structure_df,
                                                              bit_size)
    path_structure_matrix = path_files_4_ml / f'T11_x.mtx'
    path_structure_matrix_npy = path_files_4_ml / f'T11_x.npy'
    mmwrite(os.path.join(path_structure_matrix), structure_matrix, field='integer')
    np.save(path_structure_matrix_npy,structure_matrix)
    
    if args['prediction_only'] is True:
        end = time.time()
        print(f'Formatting to matrices took {end - start:.08} seconds.')
        print(f'Files in {path_files_4_ml} are ready for prediction with ML model.')
    else:
        shutil.copy(T10_activity_file_counts, path_files_4_ml)
        T9_weight_table = tuner.read_csv(T9_weight_table_file)
        T10_activity_df = tuner.read_csv(T10_activity_file)
        T9_weight_table_red = pd.DataFrame(T9_weight_table, columns=['cont_classification_task_id', 'weight']) \
            .rename(columns={"cont_classification_task_id": "task_id"}).dropna(subset=['task_id'])
        T9_weight_table_red.to_csv(path_files_4_ml / 'T9_red.csv', sep=',', index=False)
        
        activity_matrix = matrix_from_activity(T10_activity_df)
        path_activity_matrix = path_files_4_ml / f'T10_y.mtx'
        path_activity_matrix_npy = path_files_4_ml / f'T10_y.npy'
        mmwrite(os.path.join(path_activity_matrix), activity_matrix, field='integer')
        np.save(path_activity_matrix_npy, activity_matrix)

        folding_vector = folding_from_structure(T11_structure_df)
        path_structure_fold_vector = path_files_4_ml / f'T11_fold_vector'
        np.save(path_structure_fold_vector, folding_vector)
        end = time.time()
        print(f'Formatting to matrices took {end - start:.08} seconds.')
        print(f'Files in {path_files_4_ml} are ready for ML model.')
   
       



if __name__ == '__main__':
    args = vars(init_arg_parser())

    if args ['non_interactive'] is True:
        overwriting = True
    else:
        overwriting = False
    main(args, overwriting)

