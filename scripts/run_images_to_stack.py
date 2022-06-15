import argparse
import json

from punctatools.lib.convert import images_to_stacks

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameter-file', type=str, help='json file with parameters', required=True)
    parser.add_argument('-j', '--n-jobs', type=int, help='number of processes to run in parallel', default=8)
    parser.add_argument('-i', '--input-dir', type=str, default=None,
                        help='folder with images to be converted')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='folder to save results')
    args = parser.parse_args()
    parameter_file = args.parameter_file

    with open(parameter_file) as f:
        params = json.load(f)

    params['n_jobs'] = args.n_jobs
    if args.input_dir is not None:
        params['input_dir'] = args.input_dir.rstrip('/')
    if args.output_dir is not None:
        params['output_dir'] = args.output_dir.rstrip('/')

    print('\nThe following parameters will be used for conversion:')
    print(params)
    print('\n')

    images_to_stacks(parallel=True, process_name='Convert images to stacks', **params)
