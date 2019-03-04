# !/usr/bin/env python3
import os
import sys
sys.path.append('./bert/')
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import rnn_cell_impl
import bert
from bert import modeling
from bert.modeling import BertConfig, BertModel
from bert import tokenization

import fire

class MyOutputProjectionWrapper(tf.contrib.rnn.RNNCell):
    """Operator adding an output projection to the given cell.
    Note: in many cases it may be more efficient to not use this wrapper,
    but instead concatenate the whole sequence of your outputs in time,
    do the projection on this batch-concatenated sequence, then split it
    if needed or directly feed into a softmax.
    """

    def __init__(self, cell, output_size, W, activation=None, reuse=None):
        """Create a cell with output projection.
        Args:
          cell: an RNNCell, a projection to output_size is added to it.
          output_size: integer, the size of the output after projection.
          activation: (optional) an optional activation function.
          reuse: (optional) Python boolean describing whether to reuse variables
            in an existing scope.  If not `True`, and the existing scope already has
            the given variables, an error is raised.
        Raises:
          TypeError: if cell is not an RNNCell.
          ValueError: if output_size is not positive.
        """
        super(MyOutputProjectionWrapper, self).__init__(_reuse=reuse)
        rnn_cell_impl.assert_like_rnncell("cell", cell)
        if output_size < 1:
            raise ValueError(
                "Parameter output_size must be > 0: %d." % output_size)
        self._cell = cell
        self._output_size = output_size
        self._activation = activation
        self._W = W

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._output_size

    def zero_state(self, batch_size, dtype):
        with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
            return self._cell.zero_state(batch_size, dtype)

    def call(self, inputs, state):
        """Run the cell and output projection on inputs, starting from state."""
        output, res_state = self._cell(inputs, state)
        projected = tf.matmul(output, tf.transpose(self._W))
        if self._activation:
            projected = self._activation(projected)
        return projected, res_state


class ChatModelConfig:
    def __init__(self, max_x_len, max_y_len, max_decode_len,
                 vocab, config_file, ckpt_file=None, beam_width=5):
        self.max_x_len = max_x_len
        self.max_y_len = max_y_len
        self.max_decode_len = max_decode_len
        self.vocab = vocab
        self.config_file = config_file
        self.ckpt_file = ckpt_file
        self.beam_width = beam_width

class ChatModel:
    def __init__(self, chatmodel_config):
        self.chatmodel_config = chatmodel_config
        self.max_x_len = chatmodel_config.max_x_len
        self.max_y_len = chatmodel_config.max_y_len
        self.decode_max_len = chatmodel_config.max_decode_len
        self.vocab = chatmodel_config.vocab
        self.config_file = chatmodel_config.config_file
        self.ckpt_file = chatmodel_config.ckpt_file
        self.beam_width = chatmodel_config.beam_width
        self.x = tf.placeholder(tf.int32, shape=[None, self.max_x_len], name='x')
        self.x_mask = tf.placeholder(tf.int32, shape=[None, self.max_x_len], name='x_mask')
        self.x_seg = tf.placeholder(tf.int32, shape=[None, self.max_x_len], name='x_seg')
        self.x_len = tf.placeholder(tf.int32, shape=[None], name='x_len')
        self.y = tf.placeholder(tf.int32, shape=[None, self.max_y_len], name='y')
        self.y_len = tf.placeholder(tf.int32, shape=[None], name='y_len')

    def create_model(self):
        self.bert_config = BertConfig.from_json_file(self.config_file)
        self.vocab_size = self.bert_config.vocab_size
        self.hidden_size = self.bert_config.hidden_size
        self.bert_model = BertModel(config=self.bert_config,
                                    input_ids = self.x,
                                    input_mask = self.x_mask,
                                    token_type_ids = self.x_seg,
                                    is_training=True,
                                    use_one_hot_embeddings=False)
        if self.ckpt_file is not None:
            tvars = tf.trainable_variables()
            self.assignment_map, self.initialized_variable_map = modeling.get_assignment_map_from_checkpoint(
                tvars, self.ckpt_file
            )
        X = self.bert_model.get_sequence_output()
        self.embeddings = self.bert_model.get_embedding_table()
        encoder_output = X[:, 1:, :]
        encoder_state = X[:,0,:]
        batch_size = tf.shape(self.x)[0]
        start_token = tf.ones([batch_size], dtype=tf.int32)*self.vocab['<S>']
        train_output = tf.concat([tf.expand_dims(start_token, 1), self.y], 1)
        output_emb = tf.nn.embedding_lookup(self.embeddings, train_output)
        output_len = self.y_len - 1
        train_helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(
            output_emb, output_len, self.embeddings, 0.1
        )
        input_len = self.x_len - 2
        cell = tf.contrib.rnn.GRUCell(num_units=self.hidden_size)

        def decode(scope):
            with tf.variable_scope(scope):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units = self.hidden_size,
                    memory = encoder_output,
                    memory_sequence_length = input_len
                )
                attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=cell, attention_mechanism=attention_mechanism,
                    attention_layer_size = self.hidden_size
                )
                out_cell = MyOutputProjectionWrapper(
                    attention_cell, self.vocab_size, self.embeddings, reuse=False
                )
                initial_state = out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size
                )
                initial_state = initial_state.clone(
                    cell_state=encoder_state
                )
                decoder = tf.contrib.seq2seq.BasicDecoder(
                    cell=out_cell,
                    helper=train_helper,
                    initial_state=initial_state
                )
                t_final_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=decoder,
                    output_time_major=False,
                    impute_finished=True,
                    maximum_iterations=self.decode_max_len
                )
            with tf.variable_scope(scope, reuse=True):
                tiled_encoder_output = tf.contrib.seq2seq.tile_batch(
                    encoder_output, multiplier=self.beam_width
                )
                tiled_encoder_state = tf.contrib.seq2seq.tile_batch(
                    encoder_state, multiplier=self.beam_width
                )
                tiled_input_len = tf.contrib.seq2seq.tile_batch(
                    input_len, multiplier = self.beam_width
                )
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                    num_units=self.hidden_size,
                    memory=tiled_encoder_output,
                    memory_sequence_length=tiled_input_len
                )
                attention_cell = tf.contrib.seq2seq.AttentionWrapper(
                    cell=cell,
                    attention_mechanism=attention_mechanism,
                    attention_layer_size=self.hidden_size
                )
                out_cell = MyOutputProjectionWrapper(
                    attention_cell,
                    self.vocab_size,
                    self.embeddings,
                    reuse=True
                )
                initial_state = out_cell.zero_state(
                    dtype=tf.float32, batch_size=batch_size*self.beam_width
                )
                initial_state = initial_state.clone(
                    cell_state = tiled_encoder_state
                )
                self.end_token = self.vocab['<T>']
                beamDecoder = tf.contrib.seq2seq.BeamSearchDecoder(
                    cell=out_cell,
                    embedding=self.embeddings,
                    start_tokens = start_token,
                    end_token = self.end_token,
                    initial_state=initial_state,
                    beam_width = self.beam_width,
                    coverage_penalty_weight=1e-3
                )
                p_final_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                    decoder=beamDecoder,
                    output_time_major = False,
                    maximum_iterations=self.decode_max_len
                )
            return t_final_output, p_final_output

        t_output, p_output = decode('decode')

        p_output = tf.identity(
            p_output.predicted_ids[:,:,0],
            name='predictions'
        )
        return t_output, p_output

    def loss(self):
        t_output, p_output = self.create_model()
        decode_len = tf.shape(t_output.sample_id)[-1]
        mask_len = tf.maximum(decode_len, self.y_len)
        y_mask = tf.sequence_mask(
            mask_len, self.max_y_len, dtype=tf.float32
        )
        loss = tf.contrib.seq2seq.sequence_loss(
            t_output.rnn_output, self.y, weights=y_mask
        )
        p_output_sparse = self._convert_tensor_to_sparse(p_output, self.end_token)
        y_output_sparse = self._convert_tensor_to_sparse(self.y, self.end_token)
        distance = tf.reduce_sum(
            tf.edit_distance(
                p_output_sparse, y_output_sparse, normalize=False
            )
        )
        return loss, distance, p_output

    def _convert_tensor_to_sparse(self, a, end_token):
        indices = tf.where(tf.not_equal(a, 0)&tf.not_equal(a, end_token))
        values = tf.gather_nd(a, indices)
        sparse_a = tf.SparseTensor(indices, values, tf.shape(a, out_type=tf.int64))
        return sparse_a

def test_ChatModel():
    print('test_ChatModel begin!')
    vocab_file = './model/chinese_L-12_H-768_A-12/vocab.txt'
    config_file = './model/chinese_L-12_H-768_A-12/bert_config.json'
    ckpt_file = './model/chinese_L-12_H-768_A-12/bert_model.ckpt'
    tokenizer = tokenization.FullTokenizer(vocab_file)
    chatmodel_config = ChatModelConfig(
        max_x_len=100,
        max_y_len=50,
        max_decode_len=50,
        vocab=tokenizer.vocab,
        config_file=config_file,
        ckpt_file=ckpt_file,
        beam_width=5
    )
    chatmodel = ChatModel(chatmodel_config)
    loss, distance, p_output = chatmodel.loss()
    print('loss:\t', loss)
    print('distance:\t', distance)
    print('predict output:\t', p_output)
    print('test_ChatModel end!')

if __name__ == '__main__':
    test_ChatModel()