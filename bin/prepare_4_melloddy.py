"""
Common code for data preparation for the MELLODDY project.

Authors: Lukas Friedrich (Merck KGaA, Jaak Simm (KUL))
Contributors:  Ansgar Schuffenhauer (Novartis), Noe Sturm (Novartis), and Wouter Heyndrickx (Jansen)
Acknowledgements: Thanks to all WP1 members for their contributions to the common code.


The common code consists of three distinct parts:
1. Standardization of structures (SMILES format) with RDKit
2. Calculation fingperints of standardized structures
3. Formatting and filtering activity data according to rules defined in the Data Preparation Manual.


"""
import argparse


import os, sys
import time


import standardize_smiles
import calculate_descriptors
import activity_data_formatting
import hash_reference_set
import csv_2_mtx

def init_arg_parser(default_reference_file):
    parser = argparse.ArgumentParser(description='Run data processing')
    parser.add_argument('-s', '--structure_file', type=str, help='path of the structure input file', required=True)

    parser.add_argument('-a', '--activity_file', type=str, help='path of the activity input file')
    parser.add_argument('-w', '--weight_table', type=str, help='path of the weight table file')
    parser.add_argument('-c', '--config_file', type=str, help='path of the config file', required=True)
    parser.add_argument('-o', '--output_dir', type=str, help='path to the generated output directory', required=True)
    parser.add_argument('-r', '--run_name', type=str, help='name of your current run', required=True)
    parser.add_argument('-n', '--number_cpu', type=int, help='number of CPUs for calculation (default: 2 CPUs)',
                        default=2)
    parser.add_argument('-rh', '--ref_hash', type=str,
                    help='path to the reference hash key file provided by the consortium. (ref_hash.json)')
    parser.add_argument('-ni','--non_interactive', help='Enables an non-interactive mode for cluster/server usage', action='store_true')

    parser.add_argument('-p', '--prediction_only',
                        help='Preprocess only chemical structures for prediction mode', action='store_true')
    parser.add_argument('-rs', '--reference_set', type=str, help='path of the reference set file for unit tests', default = default_reference_file )
    args = parser.parse_args()
    if args.prediction_only is False:
        if args.activity_file is None or args.weight_table is None:
            parser.error("Processing for training requires activity file T4 and weight table T3.")
    return args


if __name__ == '__main__':
    start = time.time()
    main_location = os.path.dirname(os.path.realpath(__file__))
    default_reference_file = os.path.join(main_location, '../tests/structure_preparation_test/reference_set.csv')
    args = vars(init_arg_parser(default_reference_file))
    output_dir = args['output_dir']
    run_name = args['run_name']
    if args['non_interactive'] is True:
        overwriting = True
        print(f'All files for run name {run_name} will be overwritten.')
    else:
        
        override = input(f'All existing files in  {output_dir}{run_name} will be overwritten.\n Do you want to overwrite all files? (type y or yes) \n Do you want to check overwriting for each step individually? (type n or no) \n The script will be aborted if you type anything else. ')
        if override == 'y' or override=='yes':
            print(f'All files for run name {run_name} will be overwritten.')
            overwriting = True
        elif override == 'n' or override == 'no':
            print(f'Every step will ask for overwriting the files for run name {run_name}.')
            overwriting = False
        else:
            print('Processing aborted. Please change the run name and re-run the script.')
            quit()
    print('Start data preparation for the MELLODDY project.')
    standardize_smiles.main(args, overwriting)
    calculate_descriptors.main(args, overwriting)
    if args['prediction_only'] is True:
        print('Input data processed and ready for predictions with a machine-learning model.')
       
    else:
        activity_data_formatting.main(args, overwriting)
        print('Input data processed.')
    print('Hashing reference set for unit testing.')
    hash_reference_set.main(args)
    print('Processed data is converted in a ML-ready format.')
    csv_2_mtx.main(args, overwriting)
    print(f'Data preparation finished after {time.time() - start:.08} seconds.')

