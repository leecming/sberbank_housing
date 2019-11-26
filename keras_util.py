""" General Keras helper functions """
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import Callback


class ExponentialMovingAverage(Callback):
    """create a copy of trainable weights which gets updated at every
       batch using exponential weight decay.
       At end of every batch, the update is as follows:
       mv_weight -= (1 - decay) * (mv_weight - weight)
       where weight and mv_weight is the ordinal model weight and the moving
       averaged weight respectively. At the end of the training, the moving
       averaged weights are transferred to the original model.
       """

    def __init__(self, decay=0.999):
        self.decay = decay
        self.sym_trainable_weights = None  # trainable weights of model
        self.mv_trainable_weights_vals = None  # moving averaged values
        super(ExponentialMovingAverage, self).__init__()

    def on_train_begin(self, logs={}):
        self.sym_trainable_weights = self.model.trainable_weights
        # Initialize moving averaged weights using original model values
        self.mv_trainable_weights_vals = {x.name: K.get_value(x) for x in
                                          self.sym_trainable_weights}

    def on_batch_end(self, batch, logs={}):
        for weight in self.sym_trainable_weights:
            old_val = self.mv_trainable_weights_vals[weight.name]
            self.mv_trainable_weights_vals[weight.name] -= \
                (1.0 - self.decay) * (old_val - K.get_value(weight))

    def on_train_end(self, logs={}):
        for weight in self.sym_trainable_weights:
            K.set_value(weight, self.mv_trainable_weights_vals[weight.name])
