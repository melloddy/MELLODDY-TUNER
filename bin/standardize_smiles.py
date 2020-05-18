
import os

import argparse
import logging
import time
from pathlib import Path

import melloddy_tuner as tuner



def init_arg_parser():
    parser = argparse.ArgumentParser(description='Run data processing')

    parser.add_argument('-s', '--structure_file', type=str, help='path of the structure input file', required=True)
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
    # start timing
    start = time.time()
    
    output_dir = Path(args['output_dir'])
    output_dir.mkdir(exist_ok=True)
    run_name = args['run_name']
    num_cpu = args['number_cpu']

    if num_cpu < 1:
        print('Please use a positive number of CPUs.')
        quit()
    config_file = Path(args['config_file'])
    if config_file.is_file() is False:
        print('Config file does not exist.')
        quit()

    tuner.config.parameters.get_parameters(path=config_file)
    if process_reference_set is True:
        structure_file = Path(args['reference_set'])
        dir_out_run = output_dir / run_name
        dir_out_run.mkdir(exist_ok=True)
        dir_run_name = dir_out_run / 'reference_set'

    else:
        structure_file = Path(args['structure_file'])
        dir_run_name = output_dir / run_name
        
    if structure_file.is_file() is False:
        print('Structure file does not exist.')
        quit()
    
    s_filename = structure_file.stem
    dir_run_name.mkdir(exist_ok=True)
    path_results_extern = dir_run_name / 'results'
    path_results_extern.mkdir(exist_ok=True)
    path_results_intern = dir_run_name / 'results_tmp'
    path_results_intern.mkdir(exist_ok=True)
    output_dir_standardization =  path_results_intern / 'standardization'
    output_dir_standardization.mkdir(exist_ok=True)
    if overwriting is False:  
        if  os.listdir(output_dir_standardization):
            override = input(f'Do you want to override files in {output_dir_standardization}? (type y or Y) \n The script will be aborted if you type anything else. ')
            if override == 'y' or override=='Y':
                 print(f'Files for run name {run_name} will be overwritten.')
            else:
                print('Processing aborted. Please change the run name and re-run the script.')
                quit()

    # Configure the log file
    log_file_path = output_dir_standardization / f'log_standardization.log'
    logging.basicConfig(filename=log_file_path, filemode='w', format='', level=logging.ERROR)
    print(f'Start processing run name {run_name}.')
    ####################################
    # read input file
    structure_data = tuner.read_csv(structure_file)
    input_file_len = len(structure_data)
    if input_file_len == 0:
        print('Structure input is empty. Please provide a suitable structure file.')
        quit()
    ####################################
    # standardize structures with RDKit
    print('Start standardizing molecules.')
    smiles_standardized = tuner.run_standardize(structure_data['smiles'], num_cpu)
    # formatting data to output dataframes
    df_failed_smi = tuner.output_failed_smiles(smiles_standardized, structure_data)

    df_processed_smi = tuner.output_processed_smiles(smiles_standardized, structure_data)

    ####################################

    ####################################
    # write output file as csv file
    col_to_keep = ['input_compound_id', 'canonical_smiles']
    output_path = output_dir_standardization / 'T2_standardized.csv'
    output_failed_mol_path = output_dir_standardization / 'T2_failed_mols.csv'
    df_processed_smi[col_to_keep].to_csv(output_path, sep=',', index=False)
    df_failed_smi[['input_compound_id', 'smiles']].to_csv(output_failed_mol_path, sep=',', index=False)

    print(
        f'Overall processing time of {len(structure_data.index)}/{input_file_len}  molecules: {time.time() - start:.08} seconds.')
    print(f'Structure standardization of run name {run_name} done.')
        ####################################


if __name__ == '__main__':
    args = vars(init_arg_parser())

    if args ['non_interactive'] is True:
        overwriting = True
    else:
        overwriting = False
    main(args, overwriting)

