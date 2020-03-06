# std lib imports
from typing import Dict

# external libs
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.ERROR)


class SequenceToVector(models.Model):
    """
    It is an abstract class defining SequenceToVector enocoder
    abstraction. To build you own SequenceToVector encoder, subclass
    this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    """

    def __init__(self,
                 input_dim: int) -> 'SequenceToVector':
        super(SequenceToVector, self).__init__()
        self._input_dim = input_dim

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> Dict[str, tf.Tensor]:
        """
        Forward pass of Main Classifier.

        Parameters
        ----------
        vector_sequence : ``tf.Tensor``
            Sequence of embedded vectors of shape (batch_size, max_tokens_num, embedding_dim)
        sequence_mask : ``tf.Tensor``
            Boolean tensor of shape (batch_size, max_tokens_num). Entries with 1 indicate that
            token is a real token, and 0 indicate that it's a padding token.
        training : ``bool``
            Whether this call is in training mode or prediction mode.
            This flag is useful while applying dropout because dropout should
            only be applied during training.

        Returns
        -------
        An output dictionary consisting of:
        combined_vector : tf.Tensor
            A tensor of shape ``(batch_size, embedding_dim)`` representing vector
            compressed from sequence of vectors.
        layer_representations : tf.Tensor
            A tensor of shape ``(batch_size, num_layers, embedding_dim)``.
            For each layer, you typically have (batch_size, embedding_dim) combined
            vectors. This is a stack of them.
        """
        # ...
        # return {"combined_vector": combined_vector,
        #         "layer_representations": layer_representations}
        raise NotImplementedError


class DanSequenceToVector(SequenceToVector):
    """
    It is a class defining Deep Averaging Network based Sequence to Vector
    encoder. You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this DAN encoder.
    dropout : `float`
        Token dropout probability as described in the paper.
    """

    def __init__(self, input_dim: int, num_layers: int, dropout: float = 0.2):
        super(DanSequenceToVector, self).__init__(input_dim)
        self._input_dim = input_dim
        # TODO(students): start
        self._num_layers = num_layers
        self.layers_list = []

        #Add Dense layers for Feed Forward NN
        for num in range(0, self._num_layers):
            self.layers_list.append(layers.Dense(input_dim, use_bias=True, activation="relu"))

        self._dropout = dropout
        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start

        #Expand dim from [64,209] to [64,209,1] for element wise multiplication
        sequence_mask = tf.expand_dims(sequence_mask, axis=2)

        final_mask = sequence_mask

        #Dropout implementation uniform distribution ---> Bernoulli's Trials
        if training:
            dropout = tf.random.uniform(tf.shape(sequence_mask), minval=0, maxval=1)
            dropout_mask = tf.where(tf.greater_equal(dropout, self._dropout), 1.0, 0.0)
            final_mask = tf.math.multiply(final_mask, dropout_mask)

        #Mask the vector sequence for padding (Train/Predict) and dropout(Only Training)
        masked_vector_sequence = tf.math.multiply(vector_sequence, final_mask)

        #Average vector as an input for DAN
        average_vector = tf.math.reduce_mean(masked_vector_sequence, axis=1)

        # Apply the input to the Feed Forward NN for non linearity
        output = average_vector
        layer_outputs = []
        for layer in self.layers_list:
            output = layer(output)
            layer_outputs.append(output)

        layer_representations = tf.convert_to_tensor(layer_outputs, dtype=tf.float32)
        layer_representations = tf.transpose(layer_representations, perm=[1, 0, 2])

        return {"combined_vector": output,
                "layer_representations": layer_representations}


class GruSequenceToVector(SequenceToVector):
    """
    It is a class defining GRU based Sequence To Vector encoder.
    You have to implement this.

    Parameters
    ----------
    input_dim : ``str``
        Last dimension of the input input vector sequence that
        this SentenceToVector encoder will encounter.
    num_layers : ``int``
        Number of layers in this GRU-based encoder. Note that each layer
        is a GRU encoder unlike DAN where they were feedforward based.
    """

    def __init__(self, input_dim: int, num_layers: int):
        super(GruSequenceToVector, self).__init__(input_dim)
        # TODO(students): start
        self._num_layers = num_layers
        self.layers_list = []

        # Add GRU Encoding layers
        for layer_number in range(0, num_layers):
            self.layers_list.append(layers.GRU(self._input_dim, activation='tanh',
                                               return_sequences=True, return_state=True))

        # TODO(students): end

    def call(self,
             vector_sequence: tf.Tensor,
             sequence_mask: tf.Tensor,
             training=False) -> tf.Tensor:
        # TODO(students): start
        #


        # Apply input to GRU layers
        output = vector_sequence
        layer_representations = []

        for gru_layer in self.layers_list:
            output, sequence = gru_layer(output, mask=sequence_mask)
            layer_representations.append(sequence)

        layer_representations = tf.stack(layer_representations, axis=1)

        return {"combined_vector": sequence,
                "layer_representations": layer_representations}