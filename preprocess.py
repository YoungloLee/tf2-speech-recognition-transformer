import os

from datasets import preprocessor
from hparams import config2parser
from tqdm import tqdm


def preprocess(args, input_folders, out_dir):
    input_dir = os.path.join(out_dir, 'inputs')
    label_dir = os.path.join(out_dir, 'labels')
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    metadata = preprocessor.build_from_path(input_folders, input_dir, label_dir, args, tqdm=tqdm)
    write_metadata(metadata, out_dir, args)


def write_metadata(metadata, out_dir, args):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write('|'.join([str(x) for x in m]) + '\n')
    timesteps = sum([int(m[4]) for m in metadata])
    sr = args.sample_rate
    hours = timesteps / sr / 3600
    print('Write {} utterances, {} audio timesteps, ({:.2f} hours)'.format(len(metadata), timesteps, hours))
    print('Max audio timesteps length: {:.2f} secs'.format((max(m[4] for m in metadata)) / sr, ))


def norm_data(args):
    print('Selecting data folders..')
    supported_datasets = ['Kspon']
    if args.dataset not in supported_datasets:
        raise ValueError('dataset value entered {} does not belong to supported datasets: {}'.format(
            args.dataset, supported_datasets))
    if args.dataset.startswith('Kspon'):
        return [os.path.join('/mnt/Data/KsponSpeech')]


def run_preprocess(args):
    input_folders = norm_data(args)
    output_folder = os.path.join(args.base_dir, args.output)

    preprocess(args, input_folders, output_folder)


def main():
    print('initializing preprocessing..')
    args = config2parser(data='train')
    run_preprocess(args)


if __name__ == '__main__':
    main()
