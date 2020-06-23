import argparse
from multiprocessing import cpu_count
from text_syllable import token_index


def config2parser(model='Transformer', data='train'):
	parser = argparse.ArgumentParser()
	# Preprocess arguments
	parser.add_argument('--dataset', default='kspon')
	parser.add_argument('--token_style', default='syllable')
	num_classes = len(token_index)
	parser.add_argument('--num_classes', type=int, default=num_classes)
	parser.add_argument('--cleaners', default='korean_cleaners')
	if data == 'train':
		parser.add_argument('--output', default='training_data')
	elif data == 'valid':
		parser.add_argument('--output', default='validation_data')
	elif data == 'test':
		parser.add_argument('--output', default='test_data')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())

	# Train arguments
	parser.add_argument('--base_dir', default='')
	parser.add_argument('--training_input', default='training_data/train.txt')
	parser.add_argument('--validation_input', default='validation_data/valid.txt')
	parser.add_argument('--name', help='Name of logging directory.')
	parser.add_argument('--model', default=model)
	parser.add_argument('--input_dir', default='training_data', help='folder to contain inputs sentences/targets')
	parser.add_argument('--output_dir', default='output', help='folder to contain prediction')
	parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
	parser.add_argument('--eval_interval', type=int, default=1, help='epoch between eval on test data')
	parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
	parser.add_argument('--print_interval', type=int, default=5, help='steps between log to be shown')
	parser.add_argument('--train_epochs', type=int, default=200, help='total number of  training steps')
	parser.add_argument('--curriculum_limits', type=int, default=1)
	parser.add_argument('--tf_log_level', type=int, default=0, help='Tensorflow C++ log level.')
	parser.add_argument('--slack_url', default=None, help='slack webhook notification destination link')

	# Hyper-parameters
	# Audio
	parser.add_argument('--num_mels', type=int, default=128, help='number of mel-spectrogram channels')
	parser.add_argument('--rescale', type=bool, default=True, help='whether to rescale audio prior to preprocessing')
	parser.add_argument('--rescaling_max', type=float, default=0.999, help='rescaling value')
	parser.add_argument('--trim_silence', type=bool, default=True, help='whether to clip silence in Audio (at beginning and end of audio only, not the middle')
	parser.add_argument('--trim_top_db', type=int, default=30)
	parser.add_argument('--trim_fft_size', type=int, default=512)
	parser.add_argument('--trim_hop_size', type=int, default=128)
	parser.add_argument('--clip_mels_length', type=bool, default=True, help='for cases of OOM (Not really recommended, working on a workaround)')
	parser.add_argument('--use_lws', type=bool, default=False)
	# Mel spectrogram
	parser.add_argument('--n_fft', type=int, default=1024, help='extra window size is filled with 0 paddings to match this parameter')
	parser.add_argument('--hop_size', type=int, default=160, help='for 16000, 160 ~= 10 ms')
	parser.add_argument('--win_size', type=int, default=512, help='for 16000, 320 ~= 20 ms (If None, win_size = n_fft)')
	parser.add_argument('--sample_rate', type=int, default=16000, help='16000 Hz')
	parser.add_argument('--frame_shift_ms', default=None)
	# Mel and Linear spectrograms normalization/scaling and clipping
	parser.add_argument('--signal_normalization', type=bool, default=True)
	parser.add_argument('--allow_clipping_in_normalization', type=bool, default=True, help='only relevant if mel_normalization = True')
	parser.add_argument('--symmetric_mels', type=bool, default=False, help='whether to scale the data to be symmetric around 0')
	parser.add_argument('--max_abs_value', type=float, default=4., help='max absolute value of data. If symmetric, data will be [-max, max] else [0, max]')
	# Spectrogram Pre-Emphasis
	parser.add_argument('--preemphasize', type=bool, default=True, help='whether to apply filter')
	parser.add_argument('--preemphasis', type=float, default=0.97, help='filter coefficient')
	# Limits
	parser.add_argument('--min_level_db', type=int, default=-100)
	parser.add_argument('--ref_level_db', type=int, default=20)
	parser.add_argument('--fmin', type=int, default=55, help='set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])')
	parser.add_argument('--fmax', type=int, default=7600)

	if 'Transformer' in model:
		parser.add_argument('--num_layers', type=int, default=4)
		parser.add_argument('--d_model', type=int, default=256)
		parser.add_argument('--dff', type=int, default=1024)
		parser.add_argument('--num_heads', type=float, default=4)
		parser.add_argument('--dropout_rate', type=float, default=0.1)
		parser.add_argument('--lfr_m', type=int, default=4)
		parser.add_argument('--lfr_n', type=int, default=3)
		parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0)

	# Training
	parser.add_argument('--random_seed', type=int, default=777)
	parser.add_argument('--batch_size', type=int, default=16)
	parser.add_argument('--reg_weight', type=float, default=1e-6)
	parser.add_argument('--data_random_state', type=int, default=777)
	parser.add_argument('--decay_learning_rate', type=bool, default=True)
	parser.add_argument('--start_decay', type=int, default=10)
	parser.add_argument('--decay_steps', type=int, default=10)
	parser.add_argument('--decay_rate', type=float, default=0.6)
	parser.add_argument('--initial_learning_rate', type=float, default=1e-3)
	parser.add_argument('--final_learning_rate', type=float, default=1e-5)
	parser.add_argument('--adam_beta1', type=float, default=0.9)
	parser.add_argument('--adam_beta2', type=float, default=0.98)
	parser.add_argument('--adam_epsilon', type=float, default=1e-9)
	parser.add_argument('--moving_average_decay', type=float, default=0.99)
	parser.add_argument('--clip_gradients', type=float, default=400)
	args = parser.parse_args()
	return args
