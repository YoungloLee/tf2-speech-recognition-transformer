import os
import time
import datetime
import random

import infolog
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from modules.models import create_model
from modules.utils import ValueWindow
from modules.lr_scheduler import WarmUpSchedule
from modules.models.transformer import create_masks
from modules.loss import loss_function, label_error_rate
from modules.step import valid_step_teacher_forcing
from modules.utils import plot_alignment
from text_syllable import index_token
import matplotlib.pyplot as plt

log = infolog.log


def time_string():
    return datetime.datetime.now().strftime('%Y%m%d-%H%M%S')


def train(log_dir, args):
    save_dir = os.path.join(log_dir, 'pretrained')
    valid_dir = os.path.join(log_dir, 'valid-dir')
    tensorboard_dir = os.path.join(log_dir, 'events', time_string())
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, 'model')
    input_path = os.path.join(args.base_dir, args.training_input)
    valid_path = os.path.join(args.base_dir, args.validation_input)

    log('Checkpoint path: {}'.format(checkpoint_path))
    log('Loading training data from: {}'.format(input_path))
    log('Using model: {}'.format(args.model))

    # Start by setting a seed for repeatability
    tf.random.set_seed(args.random_seed)

    # To find out which devices your operations and tensors are assigned to
    tf.debugging.set_log_device_placement(False)

    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    # Set up data feeder
    from datasets.feeder import dataset
    train_dataset, valid_dataset, train_steps, valid_steps = dataset(input_path, args)

    # Track the model
    train_summary_writer = tf.summary.create_file_writer(tensorboard_dir)
    valid_summary_writer = tf.summary.create_file_writer(tensorboard_dir)

    # metrics to measure the loss of the model
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_ler = tf.keras.metrics.Mean(name='train_ler')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_acc')
    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_ler = tf.keras.metrics.Mean(name='valid_ler')
    valid_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_acc')

    # Set up model
    speech_model = create_model(args.model, save_dir, args)

    # Optimizer
    learning_rate = WarmUpSchedule(args.d_model)
    opt = Adam(learning_rate=learning_rate, beta_1=args.adam_beta1, beta_2=args.adam_beta2, epsilon=args.adam_epsilon)

    temp_learning_rate = WarmUpSchedule(args.d_model, int(train_steps * 5))
    plt.plot(temp_learning_rate(tf.range(50000, dtype=tf.float32)))
    plt.ylabel("Learning Rate")
    plt.xlabel("Train Step")
    plt.show()

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, transformer=speech_model.model)
    manager = tf.train.CheckpointManager(checkpoint, directory=save_dir, max_to_keep=5)

    checkpoint_state = tf.train.get_checkpoint_state(save_dir)
    if checkpoint_state and checkpoint_state.model_checkpoint_path:
        log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
        checkpoint.restore(manager.latest_checkpoint)
    else:
        log('No model to load at {}'.format(save_dir), slack=True)
        log('Starting new training!', slack=True)
    eval_best_loss = np.inf

    summary_list = list()
    speech_model.model.summary(line_length=180, print_fn=lambda x: summary_list.append(x))
    for summary in summary_list:
        log(summary)

    # Book keeping
    patience_count = 0
    time_window = ValueWindow(100)

    train_step_signature = [tf.TensorSpec(shape=(None, None, args.num_mels*args.lfr_m), dtype=tf.float32),
                            tf.TensorSpec(shape=(None, None), dtype=tf.int32)]

    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            predictions, _ = speech_model.model(inp, tar_inp,
                                                True,
                                                enc_padding_mask,
                                                combined_mask,
                                                dec_padding_mask)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, speech_model.model.trainable_variables)
        opt.apply_gradients(zip(gradients, speech_model.model.trainable_variables))

        tar_weight = tf.cast(tf.logical_not(tf.math.equal(tar_real, 0)), tf.int32)
        tar_len = tf.reduce_sum(tar_weight, axis=-1)
        ler = label_error_rate(tar_real, predictions, tar_len)

        train_loss(loss)
        train_ler(ler)
        train_acc(tar_real, predictions, sample_weight=tar_weight)

    @tf.function(input_signature=train_step_signature)
    def train_step_non_teacher(inp, tar):
        loss = 0.
        output = tf.expand_dims(tar[:, 0], axis=1)
        with tf.GradientTape() as tape:
            for t in range(1, tf.shape(tar)[1]):
                with tape.stop_recording():
                    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, output)
                predictions, _ = speech_model.model(inp, output,
                                                    True,
                                                    enc_padding_mask,
                                                    combined_mask,
                                                    dec_padding_mask)

                loss += loss_function(tar[:, t], predictions[:, -1, :])
                train_acc(tar[:, t], predictions[:, -1, :])
                # select the last word from the seq_len dimension
                predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)
                predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
                # concatentate the predicted_id to the output which is given to the decoder
                # as its input.
                output = tf.concat([output, predicted_id], axis=-1)

        batch_loss = (loss / tf.cast(tf.shape(tar)[1] - 1, dtype=tf.float32))
        gradients = tape.gradient(batch_loss, speech_model.model.trainable_variables)
        opt.apply_gradients(zip(gradients, speech_model.model.trainable_variables))

        tar_len = tf.reduce_sum(tf.cast(tf.logical_not(tf.math.equal(tar[:, 1:], 0)), tf.int32), axis=-1)
        ler = label_error_rate(tar[:, 1:], predictions, tar_len)

        train_loss(batch_loss)
        train_ler(ler)

    log('Speech Recognition training set to a maximum of {} epochs'.format(args.train_epochs))

    # Train
    for epoch in range(args.train_epochs):
        # show the current epoch number
        log("[INFO] starting epoch {}/{}...".format(1 + epoch, args.train_epochs))
        epochStart = time.time()

        train_loss.reset_states()
        train_ler.reset_states()
        train_acc.reset_states()
        valid_loss.reset_states()
        valid_ler.reset_states()
        valid_acc.reset_states()

        # loop over the data in batch size increments
        for (batch, (input, label)) in enumerate(train_dataset):
            start_time = time.time()
            # take a step
            use_teacher_forcing = True if random.random() < args.teacher_forcing_ratio else False
            if use_teacher_forcing:
                train_step(input, label)
            else:
                train_step_non_teacher(input, label)
            # book keeping
            time_window.append(time.time() - start_time)
            message = '[Epoch {:.3f}] [Step {:7d}] [{:.3f} sec/step, loss={:.5f}, ler={:.5f}, acc={:.5f}]'.format(
                epoch + (batch / train_steps), int(checkpoint.step), time_window.average,
                train_loss.result(), train_ler.result(), train_acc.result())

            log(message)
            checkpoint.step.assign_add(1)

            if train_loss.result() > 1e15 or np.isnan(train_loss.result()):
                log('Loss exploded to {:.5f} at step {}'.format(train_loss.result(), int(checkpoint.step)))
                raise Exception('Loss exploded')

            if int(checkpoint.step) % 1000 == 0:
                with train_summary_writer.as_default():
                    tf.summary.scalar('train_loss', train_loss.result(), step=int(checkpoint.step))
                    tf.summary.scalar('train_ler', train_ler.result(), step=int(checkpoint.step))
                    tf.summary.scalar('train_acc', train_acc.result(), step=int(checkpoint.step))

        if (1 + epoch) % args.eval_interval == 0:
            # Run eval and save eval stats
            log('\nRunning evaluation ({} steps) at step {}'.format(valid_steps, int(checkpoint.step)))
            for (batch, (input, label)) in enumerate(valid_dataset):
                # take a step
                valid_logit, align = valid_step_teacher_forcing(input, label, speech_model, valid_loss, valid_ler, valid_acc)
                if batch % (valid_steps // 10) == 0:
                    decoded = np.argmax(valid_logit, axis=-1)

                    decoded = ''.join([index_token[x] for x in decoded])
                    original = ''.join([index_token[x] for x in label.numpy()[0][1:]])

                    log('Original: %s' % original)
                    log('Decoded: %s' % decoded)
                    for num, head in enumerate(align):
                        plot_alignment(head.T, os.path.join(valid_dir, 'step-{}-align-{}-head-{}.png'.format(int(checkpoint.step), batch, num)))

            log('Eval loss & ler & acc for global step {}: {:.3f}, {:.3f}, {:.3f}'.format(
                int(checkpoint.step), valid_loss.result(), valid_ler.result(), valid_acc.result()))

            with valid_summary_writer.as_default():
                tf.summary.scalar('valid_loss', valid_loss.result(), step=int(checkpoint.step))
                tf.summary.scalar('valid_ler', valid_ler.result(), step=int(checkpoint.step))
                tf.summary.scalar('valid_acc', valid_acc.result(), step=int(checkpoint.step))

            # Save model and current global step
            save_path = manager.save()
            log("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))

            if valid_loss.result() < eval_best_loss:
                # Save model and current global step
                save_path = manager.save()
                log("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))
                log('Validation loss improved from {:.2f} to {:.2f}'.format(eval_best_loss, valid_loss.result()))
                eval_best_loss = valid_loss.result()
                patience_count = 0
            else:
                patience_count += 1
                log('Patience: {} times'.format(patience_count))
                if patience_count == args.patience:
                    log('Validation loss has not been improved for {} times, early stopping'.format(
                        args.patience))
                    log('Training complete after {} global steps!'.format(int(checkpoint.step)), slack=True)
                    return save_dir

            elapsed = (time.time() - epochStart) / 60.0
            log("one epoch took {:.4} minutes".format(elapsed))

    log('Separation training complete after {} epochs!'.format(args.train_epochs), slack=True)

    return save_dir


def sr_train(args, log_dir):
    return train(log_dir, args)
