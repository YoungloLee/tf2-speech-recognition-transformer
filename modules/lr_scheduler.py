import tensorflow as tf


class WarmUpSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=1000000):
        super(WarmUpSchedule, self).__init__()
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


class LearningRateScheduler(tf.keras.optimizers.schedules.ExponentialDecay):
    def __init__(self, initial_learning_rate, decay_steps, decay_rate, start_decay, final_learning_rate, name=None):
        super(LearningRateScheduler, self).__init__(initial_learning_rate, decay_steps, decay_rate)
        self.start_decay = start_decay
        self.final_learning_rate = final_learning_rate

    def __call__(self, step):
        with tf.name_scope(self.name or "LearningRateScheduler") as name:
            initial_learning_rate = tf.convert_to_tensor(self.initial_learning_rate, name="initial_learning_rate")
            dtype = initial_learning_rate.dtype
            decay_steps = tf.cast(self.decay_steps, dtype)
            decay_rate = tf.cast(self.decay_rate, dtype)

            step = step - self.start_decay
            global_step_recomp = tf.cast(step, dtype) * tf.cast((step > 0), dtype)
            p = global_step_recomp // decay_steps
            if self.staircase:
                p = tf.floor(p)
            lr = tf.multiply(initial_learning_rate, tf.pow(decay_rate, p), name=name)
        return tf.minimum(tf.maximum(lr, self.final_learning_rate), self.initial_learning_rate)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'final_learning_rate': self.final_learning_rate,
            'start_decay': self.start_decay
        })
        return config


def learning_rate_decay(args, steps_per_epoch):
    #################################################################
    # Narrow Exponential Decay:

    # Phase 1: lr = 1e-3
    # We only start learning rate decay after 50k steps

    # Phase 2: lr in ]1e-5, 1e-3[
    # decay reach minimal value at step 310k

    # Phase 3: lr = 1e-5
    # clip by minimal learning rate value (step > 310k)
    #################################################################

    # Compute natural exponential decay
    lr = LearningRateScheduler(args.initial_learning_rate,
                               args.decay_steps * steps_per_epoch,
                               args.decay_rate,
                               args.start_decay * steps_per_epoch,
                               args.final_learning_rate,
                               name='lr_exponential_decay')

    # clip learning rate by max and min values (initial and final values)
    return lr
