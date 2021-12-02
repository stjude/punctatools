import argparse

from punctatools.lib.convert import images_to_stacks
from punctatools.lib.utils import load_parameters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameter-file', type=str, help='json file with parameters', required=True)
    args = parser.parse_args()
    parameter_file = args.parameter_file

    param_keys = ['channel_code', 'z_position_code', 'spacing', 'n_jobs']
    param_matches = dict(input_dir='raw_dir', output_dir='converted_data_dir')
    kwargs = load_parameters(vars(), param_keys, param_matches)

    print('\nThe following are the parameters that will be used:')
    print(kwargs)
    print('\n')

    images_to_stacks(parallel=True, process_name='Convert images to stacks', **kwargs)
