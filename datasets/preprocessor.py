import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from glob import glob
import csv

import numpy as np
from text_jamo.korean import normalize
from datasets import audio


def find_files(directory, pattern='**/*.pcm'):
    return sorted(glob(os.path.join(directory, pattern), recursive=True))


def bracket_filter(sentence):
    new_sentence = str()
    flag = False
    for ch in sentence:
        if ch == '(' and flag == False:
            flag = True
            continue
        if ch == '(' and flag == True:
            flag = False
            continue
        if ch != ')' and flag == False:
            new_sentence += ch
    return new_sentence


def special_filter(sentence):
    SENTENCE_MARK = ['?', '!', '.']
    NOISE = ['o', 'n', 'u', 'b', 'l']
    EXCEPT = ['/', '+', '*', '-', '@', '$', '^', '&', '[', ']', '=', ':', ';', ',', '~']
    import re
    new_sentence = str()

    for idx, ch in enumerate(sentence):
        if ch not in SENTENCE_MARK:
            # o/, n/ 등 처리
            if idx + 1 < len(sentence) and ch in NOISE and sentence[idx + 1] == '/':
                continue

        if ch == '#':
            new_sentence += '샾'
        elif ch not in EXCEPT:
            new_sentence += ch

    pattern = re.compile(r'\s\s+')
    new_sentence = re.sub(pattern, ' ', new_sentence.strip())
    return new_sentence


def sentence_filter(raw_sentence):
    return special_filter(bracket_filter(raw_sentence))


def build_from_path(input_dirs, audio_dir, label_dir, args, tqdm=lambda x: x):
    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=args.n_jobs)
    futures = []
    index = 1
    for input_dir in input_dirs:
        files = find_files(os.path.join(input_dir))
        for wav_path in files:
            text_path = os.path.splitext(wav_path)[0] + '.txt'
            futures.append(executor.submit(partial(_process_utterance, audio_dir, label_dir, index, wav_path, text_path, args)))
            index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(audio_dir, label_dir, index, wav_path, text_path, args):
    """
    Preprocesses a single utterance wav/text_jamo pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - linear_dir: the directory to write the linear spectrograms into
        - wav_dir: the directory to write the preprocessed wav into
        - index: the numeric index to use in the spectogram filename
        - wav_path: path to the audio file containing the speech input
        - text_jamo: text_jamo spoken in the input audio file
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text_jamo)
    """
    try:
        # Load the audio as numpy array
        # wav = audio.load_wav(wav_path, sr=args.sample_rate)
        with open(wav_path, 'rb') as pcmfile:
            buf = pcmfile.read()
            wav = np.frombuffer(buf, dtype='int16')
    except FileNotFoundError: #catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
        return None

    # rescale wav
    if args.rescale:
        wav = wav / np.abs(wav).max() * args.rescaling_max

    # M-AILABS extra silence specific
    if args.trim_silence:
        wav = audio.trim_silence(wav, args)

    # [-1, 1]
    out = wav
    constant_values = 0.
    out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav, args).astype(out_dtype)
    mel_frames = mel_spectrogram.shape[1]

    # Ensure time resolution adjustement between audio and mel-spectrogram
    pad = audio.librosa_pad_lr(wav, args.n_fft, audio.get_hop_size(args))

    # Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
    out = np.pad(out, (0, pad), mode='reflect')
    assert len(out) >= mel_frames * audio.get_hop_size(args)

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(args)]
    assert len(out) % audio.get_hop_size(args) == 0
    time_steps = len(out)

    # text_jamo sequence
    with open(text_path, 'r', encoding='CP949') as f:
        line = f.readline()

    # ETRI transcription rule
    line = sentence_filter(line).upper()
    label_sequence = normalize(line)
    # print(label_sequence)

    # Write the spectrogram and audio to disk
    mel_filename = 'mel-{}.npy'.format(index)
    label_filename = 'label-{}.txt'.format(index)
    np.save(os.path.join(audio_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    with open(os.path.join(label_dir, label_filename), 'w', encoding='utf-8') as f_out:
        f_out.write(label_sequence)

    # Return a tuple describing this training example
    return (wav_path, text_path, mel_filename, label_filename, time_steps, mel_frames)


def build_from_path_clova(input_dirs, audio_dir, label_dir, args, tqdm=lambda x: x):
    # We use ProcessPoolExecutor to parallelize across processes, this is just for
    # optimization purposes and it can be omited
    executor = ProcessPoolExecutor(max_workers=args.n_jobs)
    futures = []
    index = 1
    for input_dir in input_dirs:
        files = find_files(os.path.join(input_dir), pattern='**/*.wav')
        for wav_path in files:
            text_path = os.path.splitext(wav_path)[0][:-2] + 'script.csv'
            futures.append(executor.submit(partial(_process_utterance_clova, audio_dir, label_dir, index, wav_path, text_path, args)))
            index += 1

    return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance_clova(audio_dir, label_dir, index, wav_path, text_path, args):
    """
    Preprocesses a single utterance wav/text_jamo pair

    this writes the mel scale spectogram to disk and return a tuple to write
    to the train.txt file

    Args:
        - mel_dir: the directory to write the mel spectograms into
        - linear_dir: the directory to write the linear spectrograms into
        - wav_dir: the directory to write the preprocessed wav into
        - index: the numeric index to use in the spectogram filename
        - wav_path: path to the audio file containing the speech input
        - text_jamo: text_jamo spoken in the input audio file
        - hparams: hyper parameters

    Returns:
        - A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text_jamo)
    """
    try:
        # Load the audio as numpy array
        wav = audio.load_wav(wav_path, sr=args.sample_rate)
    except FileNotFoundError: #catch missing wav exception
        print('file {} present in csv metadata is not present in wav folder. skipping!'.format(wav_path))
        return None

    # rescale wav
    if args.rescale:
        wav = wav / np.abs(wav).max() * args.rescaling_max

    # M-AILABS extra silence specific
    if args.trim_silence:
        wav = audio.trim_silence(wav, args)

    # [-1, 1]
    out = wav
    constant_values = 0.
    out_dtype = np.float32

    # Compute the mel scale spectrogram from the wav
    mel_spectrogram = audio.melspectrogram(wav, args).astype(out_dtype)
    mel_frames = mel_spectrogram.shape[1]

    # Ensure time resolution adjustement between audio and mel-spectrogram
    pad = audio.librosa_pad_lr(wav, args.n_fft, audio.get_hop_size(args))

    # Reflect pad audio signal (Just like it's done in Librosa to avoid frame inconsistency)
    out = np.pad(out, (0, pad), mode='reflect')
    assert len(out) >= mel_frames * audio.get_hop_size(args)

    # time resolution adjustment
    # ensure length of raw audio is multiple of hop size so that we can use
    # transposed convolution to upsample
    out = out[:mel_frames * audio.get_hop_size(args)]
    assert len(out) % audio.get_hop_size(args) == 0
    time_steps = len(out)

    # text_jamo sequence
    with open(text_path, 'r', encoding='utf-8', newline='') as f:
        rdr = csv.reader(f)
        for x in rdr:
            if os.path.basename(wav_path) == x[0]:
                line = x[1]

    # ETRI transcription rule
    line = sentence_filter(line).upper()
    label_sequence = normalize(line)
    print(label_sequence)

    # Write the spectrogram and audio to disk
    mel_filename = 'mel-{}.npy'.format(index)
    label_filename = 'label-{}.txt'.format(index)
    np.save(os.path.join(audio_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

    with open(os.path.join(label_dir, label_filename), 'w', encoding='utf-8') as f_out:
        f_out.write(label_sequence)

    # Return a tuple describing this training example
    return (wav_path, text_path, mel_filename, label_filename, time_steps, mel_frames)