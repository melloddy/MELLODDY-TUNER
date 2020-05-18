"""
Common code for data preparation for the MELLODDY project.

 Calculation of descriptors (fingerprints) with RDKit

"""

import os
from datetime import datetime


import argparse
import logging
import time
from pathlib import Path

import melloddy_tuner as tuner
import pandas as pd


def init_arg_parser():
    parser = argparse.ArgumentParser(description='Run Fingerprint calculation')

    parser.add_argument('-c', '--config_file', type=str, help='path of the config file', required=True)
    parser.add_argument('-o', '--output_dir', type=str, help='path to the generated output directory', required=True)
    parser.add_argument('-r', '--run_name', type=str, help='name of your current run', required=True)
    parser.add_argument('-n', '--number_cpu', type=int, help='number of CPUs for calculation (default: 2 CPUs)',
                        default=2)
    parser.add_argument('-ni','--non_interactive', help='Enables an non-interactive mode for cluster/server usage', action='store_true', default=False)

    parser.add_argument('-p', '--prediction_only',
                        help='Preprocess only chemical structures for prediction mode', action='store_true', default = False)
    args = parser.parse_args()
    return args



def main(args, overwriting = True, process_reference_set = False):

    ####################################
    # get back parameters
    start = time.time()
    if args['number_cpu'] < 1:
        print('Please use a positive number of CPUs.')
        quit()
    config_file = Path(args['config_file'])
    if config_file.is_file() is False:
        print('Config file does not exist.')
        quit()

    tuner.config.parameters.get_parameters(path=config_file)
    output_dir = Path(args['output_dir'])
    output_dir.mkdir(exist_ok=True)
    run_name = args['run_name']
    if process_reference_set is True:
        dir_run_name = output_dir / run_name / 'reference_set'
    else:
        dir_run_name = output_dir / run_name
    dir_run_name.mkdir(exist_ok=True)
    path_results_intern = dir_run_name / 'results_tmp'  
    path_results_intern.mkdir(exist_ok=True)
    path_results_extern = dir_run_name / 'results'
    path_results_extern.mkdir(exist_ok=True)
    output_dir_standardization = path_results_intern / 'standardization'
    output_dir_descriptors = path_results_intern / 'descriptors'
    output_dir_descriptors.mkdir(exist_ok=True)
    if overwriting is False:
        if  os.listdir(output_dir_descriptors):
            override = input(f'Do you want to override files in {output_dir_descriptors} ? (type y or Y) \n The script will be aborted if you type anything else. ')
            if override == 'y' or override=='Y':
                 print(f'Files for run name {run_name} will be overwritten.')
            else:
                print('Processing aborted. Please change the run name and re-run the script.')
                quit()

#             s_path = Path(args['output_dir'])
#             s_filename = structure_file.stem
    num_cpu = args['number_cpu']



    ####################################
    # read input file
    file_standardized = output_dir_standardization / 'T2_standardized.csv'

    if not file_standardized.exists():
        print('Standardized structure file was not found.')
        quit()

    structure_data = tuner.helper.read_csv(file_standardized)
    log_file_path = output_dir_descriptors / 'log_fingerprint_calc.log'
    logging.basicConfig(filename=log_file_path, filemode='w', format='', level=logging.ERROR)
    ####################################
    # calculate fingerprint
    print('Start calculating fingerprints.')

    ecfp = tuner.run_fingerprint(structure_data['canonical_smiles'], num_cpu)
    df_processed_desc = tuner.output_processed_descriptors(ecfp, structure_data)
    print('Finished fingerprint calculation.')
    ####################################
    print('Formatting fingerprints.')
    structure_data_duplicates = tuner.output_descriptor_duplicates(df_processed_desc)
    mapping_table_T5 = tuner.output_mapping_table_T5(df_processed_desc)

    # Save mapping table and fingeprint duplicates
    mapping_table_T5_path = output_dir_descriptors / 'mapping_table_T5.csv'
    duplicates_fingerprint_path = output_dir_descriptors / f'desc_duplicates.csv'
    mapping_table_T5.to_csv(mapping_table_T5_path, sep=',', index=False)
    structure_data_duplicates.to_csv(duplicates_fingerprint_path, sep=',', index=False)
    print('Fingerprint calculation done.')
    ####################################
    # CSR Matrix generation and LSH clustering
    print('Starting LSH clustering.')
    lsh_folding = tuner.LSHFolding()
    df_high_entropy_bits = lsh_folding.calc_highest_entropy_bits(ecfp)
    df_folds = lsh_folding.run_lsh_calculation(ecfp)
    ####################################
    # Format and save dataframes
    path_high_entropy_bits = output_dir_descriptors / 'desc_high_entropy_bits.csv'
    df_high_entropy_bits.to_csv(path_high_entropy_bits, sep=',', index=False)

    ####################################
    if process_reference_set is True or args['prediction_only'] is True:
        df_output_data = pd.concat([structure_data, df_folds], axis=1)
        df_output_data = df_output_data.drop_duplicates(['descriptor_vector_id', 'fp_json', 'fp_val_json']) \
            .sort_values('descriptor_vector_id')
        mappting_table_T6_path = output_dir_descriptors / 'T6_prediction_only.csv'
        df_output_data.to_csv(mappting_table_T6_path, sep=',',
                              columns=['descriptor_vector_id', 'fp_json', 'fp_val_json', 'fold_id'],
                              index=False)

        mapping_table_T10_path = output_dir_descriptors / 'mapping_table_T10__prediction_only.csv'
        df_output_data[['descriptor_vector_id', 'fold_id']].to_csv(mapping_table_T10_path, sep=',',
                                                                   index=False)

        output_data_remapped = tuner.ActivityDataFormatting.map_2_cont_id(df_output_data,
                                                                          'descriptor_vector_id').sort_values(
            'cont_descriptor_vector_id')
        if process_reference_set is True:
            path_structure_data_final_T11 = path_results_extern / 'reference_set_T11.csv'
        else:
            path_structure_data_final_T11 = path_results_extern / 'T11_prediction_only.csv'
        output_data_remapped.to_csv(path_structure_data_final_T11, sep=',',
                                    columns=['cont_descriptor_vector_id', 'descriptor_vector_id', 'fp_json',
                                             'fp_val_json', 'fold_id'],
                                    index=False)
        end = time.time()
        print(
            f'Fingerprint calculation and LSH clustering took {end - start:.08} seconds. T11 file for prediction was created.')
    else:
        # Save output files T6, T10
        df_output_data = pd.concat([structure_data, df_folds], axis=1)
        df_output_data = df_output_data.drop_duplicates(['descriptor_vector_id', 'fp_json', 'fp_val_json']) \
            .sort_values('descriptor_vector_id')

        mappting_table_T6_path = output_dir_descriptors / 'T6.csv'
        df_output_data.to_csv(mappting_table_T6_path, sep=',',
                              columns=['descriptor_vector_id', 'fp_json', 'fp_val_json', 'fold_id'],
                              index=False)

        mapping_table_T10_path = output_dir_descriptors / f'mapping_table_T10.csv'
        df_output_data[['descriptor_vector_id', 'fold_id']].to_csv(mapping_table_T10_path, sep=',',
                                                                   index=False)

        end = time.time()
        print(f'Fingerprint calculation and LSH clustering took {end - start:.08} seconds.')
        print(f'Descriptor calculation of run name {run_name} done.')

if __name__ == '__main__':
    args = vars(init_arg_parser())
    if args ['non_interactive'] is True:
        overwriting = True
    else:
        overwriting = False
    main(args, overwriting)

