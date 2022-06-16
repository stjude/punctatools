import argparse
import json
import os

from punctatools.lib.quantify import quantify_batch
from punctatools.lib.segment import segment_puncta_batch

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

    os.makedirs(params['output_dir'], exist_ok=True)
    with open(os.path.join(params['output_dir'], parameter_file.split('/')[-1]), 'w') as f:
        json.dump(params, f, indent=4)

    print('\nThe following parameters that will be used for the analysis:')
    print(params)
    print('\n')

    channel_names = params.pop('channel_names')
    puncta_segm_dir = params.pop('puncta_segm_dir')
    roi_quant_dir = params.pop('roi_quant_dir')
    puncta_quant_dir = params.pop('puncta_quant_dir')

    segm_kwargs = params.copy()
    segm_kwargs['output_dir'] = os.path.join(segm_kwargs['output_dir'], puncta_segm_dir)
    segment_puncta_batch(parallel=True, process_name='Segment puncta', **segm_kwargs)

    print('\n')

    quantify_batch(input_dir=segm_kwargs['output_dir'],
                   output_dir_puncta=os.path.join(params['output_dir'], puncta_quant_dir),
                   output_dir_roi=os.path.join(params['output_dir'], roi_quant_dir),
                   parallel=True, n_jobs=params['n_jobs'],
                   channel_names=channel_names,
                   puncta_channels=params['puncta_channels'],
                   process_name='Quantify puncta')
