# !/usr/bin/env python3

import os
import sys
sys.path.append('./bert/')
sys.path.append('./')
import tensorflow as tf
import configparser
from data import file_based_input_fn_builder
from model import ChatModelConfig, ChatModel
import bert
from bert import tokenization
from bert import optimization
from bert.optimization import create_optimizer

def make_feed_dict(model, inputs, droprate):
    dict = {
        model.x : inputs['x'],
        model.x_mask : inputs['x_mask'],
        model.x_seg : inputs['x_seg'],
        model.x_len : inputs['x_len'],
        model.y : inputs['y'],
        model.y_len : inputs['y_len']
    }
    return dict

def train():
    parser = configparser.ConfigParser()
    parser.read('params.ini')
    max_x_len = int(parser.get('chat_model', 'max_x_len'))
    max_y_len = int(parser.get('chat_model', 'max_y_len'))
    decode_max_len = int(parser.get('chat_model', 'decode_max_len'))
    vocab_file = parser.get('chat_model', 'vocab_file')
    config_file = parser.get('chat_model', 'config_file')
    ckpt_file = parser.get('chat_model', 'ckpt_file')
    beam_width = int(parser.get('chat_model', 'beam_width'))
    batch_size = int(parser.get('chat_model', 'batch_size'))
    lr = float(parser.get('chat_model', 'lr'))
    train_nums = int(parser.get('chat_model', 'train_data_size'))
    warmup_proportion = float(parser.get('chat_model', 'warmup_proportion'))
    epochs = int(parser.get('chat_model', 'epochs'))
    log_dir = parser.get('chat_model', 'log_dir')
    data_dir = parser.get('chat_model', 'data_dir')
    train_file = os.path.join(data_dir, 'train.tfrecord')
    # vocab_file = './model/chinese_L-12_H-768_A-12/vocab.txt'
    tokenizer = tokenization.FullTokenizer(vocab_file)
    chatmodel_config = ChatModelConfig(
        max_x_len, max_y_len, decode_max_len,
        tokenizer.vocab, config_file, ckpt_file, beam_width
    )
    os.makedirs(log_dir, exist_ok=True)
    graph = tf.Graph()
    step = 0
    with graph.as_default():
        input_fn = file_based_input_fn_builder(train_file, max_x_len, max_y_len, True, True)
        ds = input_fn(batch_size)
        iterator = ds.make_one_shot_iterator()
        batch_inputs = iterator.get_next()
        chat_model = ChatModel(chatmodel_config)
        loss, distance, predictions = chat_model.loss()
        num_train_steps = int(train_nums/batch_size*epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        train_op = optimization.create_optimizer(
            loss, lr, num_train_steps, num_warmup_steps, False)
        saver = tf.train.Saver()
        scaf = tf.train.Scaffold(saver=saver)
        tf.Session().run(tf.global_variables_initializer())
        with tf.train.MonitoredTrainingSession(checkpoint_dir=log_dir,
                                               scaffold=scaf,
                                               hooks=[tf.train.StopAtStepHook(last_step=num_train_steps),
                                                      tf.train.NanTensorHook(loss)],
                                               config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
            best_loss = float('inf')
            best_acc = 0
            try:
                while not sess.should_stop():
                    trainDatas = sess._tf_sess().run(batch_inputs)
                    feed_dict = make_feed_dict(chat_model, trainDatas, 0.1)
                    train_loss, _ = sess._tf_sess().run(
                        [loss, train_op], feed_dict=feed_dict
                    )
                    step += 1
                    if step % 100 == 0:
                        print('[train loss]:\t', train_loss)
                        print('[predict acc]:\t', distance)
            except KeyboardInterrupt as e:
                saver.save(sess._sess, os.path.join(log_dir, 'except_model'), global_step=tf.train.get_or_create_global_step())
            except Exception as e:
                print(e)


if __name__ == '__main__':
    train()










