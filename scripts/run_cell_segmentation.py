import argparse

from punctatools.lib.segment import segment_cells_batch
from punctatools.lib.utils import load_parameters

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--parameter-file', type=str, help='json file with parameters', required=True)
    args = parser.parse_args()
    parameter_file = args.parameter_file

    param_keys = ['diameter',
                  'model_type',
                  'do_3D',
                  'remove_small_mode',
                  'remove_small_diam_fraction',
                  'flow_threshold',
                  'cellprob_threshold',
                  'gpu']
    param_matches = dict(input_dir='converted_data_dir',
                         output_dir='cell_segmentation_dir',
                         channel='cells_channel')
    kwargs = load_parameters(vars(), param_keys, param_matches)

    print('\nThe following are the parameters that will be used:')
    print(kwargs)
    print('\n')

    segment_cells_batch(**kwargs)
