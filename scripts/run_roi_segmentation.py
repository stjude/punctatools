import argparse
import json

from punctatools.lib.segment import segment_roi_batch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameter-file', type=str, help='json file with parameters', required=True)
    parser.add_argument('-i', '--input-dir', type=str, default=None,
                        help='folder with images to be converted')
    parser.add_argument('-o', '--output-dir', type=str, default=None,
                        help='folder to save results')
    args = parser.parse_args()
    parameter_file = args.parameter_file

    with open(parameter_file) as f:
        params = json.load(f)

    if args.input_dir is not None:
        params['input_dir'] = args.input_dir.rstrip('/')
    if args.output_dir is not None:
        params['output_dir'] = args.output_dir.rstrip('/')

    print('\nThe following parameters will be used for segmentation:')
    print(params)
    print('\n')

    segment_roi_batch(**params)
