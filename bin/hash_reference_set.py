import argparse
import os, hashlib
import json
from datetime import datetime
import time
from pathlib import Path
import standardize_smiles
import calculate_descriptors

from melloddy_tuner.version import __version__

def init_arg_parser(default_reference_file):
    
    
    parser = argparse.ArgumentParser(description='Run data processing')
    parser.add_argument('-c', '--config_file', type=str, help='path of the config file', required=True)
    parser.add_argument('-o', '--output_dir', type=str, help='path to the generated output directory', required=True)
    parser.add_argument('-n', '--number_cpu', type=int, help='number of CPUs for calculation (default: 2 CPUs)',
                        default=2)
    parser.add_argument('-r', '--run_name', type=str, help='name of your current run', required=True)
    parser.add_argument('-rs', '--reference_set', type=str, help='path of the reference set file for unit tests', default = default_reference_file )
    args = parser.parse_args()
    return args


def hash_reference_dir(args):
    """
    """
    config_file = args['config_file']
    output_dir = Path(args['output_dir'])
    run_name = Path(args['run_name'])
    ref_dir = output_dir / run_name / 'reference_set'
    sha256_hash = hashlib.sha256()
    if not ref_dir.exists():
        return print('Reference set directory does not exist.')
    try:
        filepath = ref_dir / 'results/reference_set_T11.csv'
        if filepath.exists() is True:
            print('Hashing unit test file', filepath)
            with open(filepath, "rb") as f:
                # Read file in as little chunks
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(hashlib.sha256(byte_block).hexdigest().encode('utf-8'))
                f.close()
        else:
            print('Reference set file T11 not found.')

        with open(config_file, 'rb') as cfg_f:
            # Read file in as little chunks
            
            print('Hashing', config_file)
            for byte_block in iter(lambda: cfg_f.read(4096), b""):
                sha256_hash.update(hashlib.sha256(byte_block).hexdigest().encode('utf-8'))
            cfg_f.close()

        print(f'Hashing version: {__version__}')
        sha256_hash.update(hashlib.sha256(__version__.encode('utf-8')).hexdigest().encode('utf-8'))  

    except:
        import traceback
        # Print the stack traceback
        traceback.print_exc()
        return print('General Error.')
    hash_hex = sha256_hash.hexdigest()
    reference_hash = {'unit_test_hash': hash_hex}
    p_output_dir = Path(output_dir).joinpath(run_name)
    path_gen_hash = p_output_dir / ''
    path_gen_hash.mkdir(exist_ok=True)
    with open(path_gen_hash / 'generated_hash.json', 'w') as json_file:
        json.dump(reference_hash, json_file)
    return print('Done.')

def main(args):
    standardize_smiles.main(args, overwriting = True, process_reference_set =True)
    calculate_descriptors.main(args, overwriting = True, process_reference_set= True)
    hash_reference_dir(args)

if __name__ == '__main__':
    start = time.time()
    main_location = os.path.dirname(os.path.realpath(__file__))
    default_reference_file = os.path.join(main_location, '../tests/structure_preparation_test/reference_set.csv')
    args = vars(init_arg_parser(default_reference_file))
    main(args)
    print(f'Hashing reference data finished after {time.time() - start:.08} seconds.')
