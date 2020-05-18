"""
Common code for data preparation for the MELLODDY project.

Part 3: Formatting and filtering activity data

"""

import os
from datetime import datetime


import argparse
import time
from pathlib import Path

from melloddy_tuner import ActivityDataFormatting, config, read_csv


def init_arg_parser():
    parser = argparse.ArgumentParser(description='Run Fingerprint calculation')
    parser.add_argument('-a', '--activity_file', type=str, help='path of the activity input file', required=True)
    parser.add_argument('-w', '--weight_table', type=str, help='path of the weight table file', required=True)
    parser.add_argument('-c', '--config_file', type=str, help='path of the config file', required=True)
    parser.add_argument('-o', '--output_dir', type=str, help='path to the generated output directory', required=True)
    parser.add_argument('-r', '--run_name', type=str, help='name of your current run', required=True)
    parser.add_argument('-n', '--number_cpu', type=int, help='number of CPUs for calculation (default: 2 CPUs)',
                        default=2)
    parser.add_argument('-ni','--non_interactive', help='Enables an non-interactive mode for cluster/server usage', action='store_true', default=False)
    args = parser.parse_args()
    return args



def main(args, overwriting=True):
    """
    Main function reading input files, executing functions and writing output files.
    """
    ####################################
    # get parameters
    start = time.time()


    activity_file = Path(args['activity_file'])
    weight_table = Path(args['weight_table'])
    config_file = Path(args['config_file'])
    if config_file.is_file() is False:
        print('Config file does not exist.')
        quit()

    config.parameters.get_parameters(path=config_file)
    if activity_file.is_file() is False:
        print('Activity file does not exist.')
        quit()


    output_path = Path(args['output_dir'])
    output_path.mkdir(exist_ok=True)
    run_name = args['run_name']
    output_dir = output_path / run_name
    path_results_intern = output_dir / 'results_tmp'  
    path_results_intern.mkdir(exist_ok=True)
    path_results_extern = output_dir / 'results'
    path_results_extern.mkdir(exist_ok=True)
    output_dir_standardization = path_results_intern / 'standardization'
    output_dir_descriptors = path_results_intern / 'descriptors'
    output_dir_activity_data = path_results_intern / 'activity_data'
    output_dir_activity_data.mkdir(exist_ok=True)
    if overwriting is False:
        if  os.listdir(output_dir_activity_data):
            override = input(f'Do you want to override files in {output_dir_activity_data} ? (type y or Y) \n The script will be aborted if you type anything else. ')
            if override == 'y' or override=='Y':
                print(f'Files for run name {run_name} will be overwritten.')
            else:
                print('Processing aborted. Please change the run name and re-run the script.')
                quit()
    
    
    ###################################
    # read mapping table T5 and activity file
    path_mapping_table_T5 = output_dir_descriptors / 'mapping_table_T5.csv'
    path_mapping_table_T10 = output_dir_descriptors / 'mapping_table_T10.csv'
    path_T6_structure_data = output_dir_descriptors /  'T6.csv'

    # check if mapping table exists
    if path_mapping_table_T5.is_file() and path_mapping_table_T10.is_file() and path_T6_structure_data.is_file() is False:
        print(
            'Structure data file T6, or mapping table T5 or T10 was not found. Please perform first the standardization and fingerprint calculation.')
        quit()
    else:
        print('Start activity data formatting.')

        # read input files (mapping table T5, T10) activity data T4, and weight table T3
        mapping_table_T5 = read_csv(path_mapping_table_T5)
        activity_data = read_csv(activity_file)
        mapping_table_T10 = read_csv(path_mapping_table_T10)
        import pandas as pd
        pd.options.mode.chained_assignment = 'raise'
        act_data_format = ActivityDataFormatting(activity_data, mapping_table_T5, mapping_table_T10)
        del (activity_data, mapping_table_T5, mapping_table_T10)
        act_data_format.run_formatting()

        # identify and write output file for failed activity entries
        data_failed = act_data_format.filter_failed_structures()
        path_failed_data = output_dir_activity_data / 'T4_failed_structures.csv'
        data_failed.to_csv(path_failed_data, sep=',',
                           columns=['input_compound_id', 'classification_task_id', 'class_label'],
                           index=False)

        # identify duplicated id pairs and save them.
        data_duplicated_id_pairs = act_data_format.data_duplicates
        path_duplicated_structures = output_dir_activity_data / 'T4_duplicates.csv'
        data_duplicated_id_pairs.to_csv(path_duplicated_structures, sep=',',
                                        columns=['classification_task_id', 'descriptor_vector_id',
                                                 'class_label'],
                                        index=False)
        del (data_failed, data_duplicated_id_pairs)

        # save excluded data
        data_excluded = act_data_format.select_excluded_data()
        path_excluded_data = output_dir_activity_data / 'T4_excluded_data.csv'
        data_excluded.to_csv(path_excluded_data, sep=',',
                             columns=['classification_task_id', 'descriptor_vector_id', 'class_label'],
                             index=False)

        # save T11
        act_data_format.remapping_2_cont_ids()
        structure_data_T6 = read_csv(path_T6_structure_data)
        structure_data_T11 = act_data_format.make_T11(structure_data_T6).sort_values(
            'cont_descriptor_vector_id')
        path_structure_data_final_T11 = path_results_extern / 'T11.csv'
        structure_data_T11.to_csv(path_structure_data_final_T11, sep=',',
                                  columns=['cont_descriptor_vector_id', 'descriptor_vector_id', 'fp_json',
                                           'fp_val_json', 'fold_id'],
                                  index=False)
        del (structure_data_T6, structure_data_T11)

        # save T10
        data_remapped = act_data_format.data_remapped.sort_values(
            'cont_classification_task_id')
        path_data_final_T10 = path_results_extern / f'T10.csv'
        data_remapped.to_csv(path_data_final_T10, sep=',',
                             columns=['cont_descriptor_vector_id', 'cont_classification_task_id', 'class_label',
                                      'fold_id'],
                             index=False)
        # count labels per fold and save it.
        data_final_counts = act_data_format.count_labels_per_fold(data_remapped).sort_values(
            'cont_classification_task_id')
        path_final_counts = path_results_extern / 'T10_counts.csv'
        data_final_counts.to_csv(path_final_counts, sep=',',
                                 columns=['cont_classification_task_id', 'class_label', 'fold_id',
                                          'label_counts'],
                                 index=False)

        del (data_remapped, data_final_counts)

        # update weight table T3 with cont_task_ids and save reduced table T3 as T9
        weight_table_T3 = read_csv(weight_table)
        weight_table_T3_mapped = act_data_format.map_T3(weight_table_T3)
        
        path_weight_table_T3_mapped = path_results_extern / f'weight_table_T3_mapped.csv'
        weight_table_T3_mapped.to_csv(path_weight_table_T3_mapped, sep=',', index=False)
        
        path_weight_table_T9 = path_results_extern / f'weight_table_T9.csv'
        col_to_keep = ['cont_classification_task_id', 'assay_type', 'weight']
        weight_table_T3_mapped.to_csv(path_weight_table_T9, sep=',', index=False, columns=col_to_keep)

        end = time.time()
        print(f'Formatting of activity data took {end - start:.08} seconds.')
        print(f'Activity data processing of run name {run_name} done.')


if __name__ == '__main__':
    args = vars(init_arg_parser())
    if args ['non_interactive'] is True:
        overwriting = True
    else:
        overwriting = False
    main(args, overwriting)

