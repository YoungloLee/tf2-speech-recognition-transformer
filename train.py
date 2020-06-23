import os
from time import sleep

import infolog
from hparams import config2parser
from infolog import log

log = infolog.log


def save_seq(file, sequence, input_path):
    '''Save training state to disk. (To skip for future runs)
    '''
    sequence = [str(int(s)) for s in sequence] + [input_path]
    with open(file, 'w') as f:
        f.write('|'.join(sequence))


def read_seq(file):
    '''Load training state from disk. (To skip if not first run)
    '''
    if os.path.isfile(file):
        with open(file, 'r') as f:
            sequence = f.read().split('|')
        return [bool(int(s)) for s in sequence[:-1]], sequence[-1]
    else:
        return [0], ''


def prepare_run(args):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'Terminal_train_log'), run_name, args.slack_url)
    return log_dir


def train(args, log_dir):
    state_file = os.path.join(log_dir, 'state_log')
    # Get training states
    state, input_path = read_seq(state_file)

    log('\n#############################################################\n')
    log('Speech Recognition Train\n')
    log('#############################################################\n')
    if args.model == 'Transformer':
        from modules.train import sr_train
        checkpoint = sr_train(args, log_dir)
    # Sleep 1/2 second to let previous graph close
    sleep(0.5)
    if checkpoint is None:
        raise ValueError('Error occured while training, Exiting!')

    state = 1

    if state:
        log('TRAINING IS ALREADY COMPLETE!!')
 

def main():
    model = 'Transformer'
    accepted_models = ['Transformer']
    args = config2parser(model)
    if args.model not in accepted_models:
        raise ValueError('please enter a valid model to train: {}'.format(accepted_models))
    log_dir = prepare_run(args)
    train(args, log_dir)


if __name__ == '__main__':
    main()
