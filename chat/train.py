# !/usr/bin/env python3

'''
note: if run on ubuntu, do following code:
export LC_CTYPE=C.UTF-8
'''

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
    dropout_rate = float(parser.get('chat_model', 'dropout_rate'))
    train_nums = int(parser.get('chat_model', 'train_data_size'))
    warmup_proportion = float(parser.get('chat_model', 'warmup_proportion'))
    epochs = int(parser.get('chat_model', 'epochs'))
    length_penalty_weight = float(parser.get('chat_model', 'length_penalty_weight'))
    coverage_penalty_weight = float(parser.get('chat_model', 'coverage_penalty_weight'))
    log_dir = parser.get('chat_model', 'log_dir')
    data_dir = parser.get('chat_model', 'data_dir')
    train_file = os.path.join(data_dir, 'train.tfrecord')
    # vocab_file = './model/chinese_L-12_H-768_A-12/vocab.txt'
    tokenizer = tokenization.FullTokenizer(vocab_file)
    chatmodel_config = ChatModelConfig(
        max_x_len, max_y_len, decode_max_len,
        tokenizer.vocab, config_file, dropout_rate, ckpt_file, beam_width,
        coverage_penalty_weight, length_penalty_weight
    )
    os.makedirs(log_dir, exist_ok=True)
    graph = tf.Graph()
    step = 0
    eval_log = []
    with graph.as_default():
        input_fn = file_based_input_fn_builder(train_file, max_x_len, max_y_len, True, True)
        ds = input_fn(batch_size)
        iterator = ds.make_one_shot_iterator()
        batch_inputs = iterator.get_next()
        chat_model = ChatModel(chatmodel_config)
        loss, distance, predictions, train_predictions = chat_model.loss()
        num_train_steps = int(train_nums/batch_size*epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)
        train_op = optimization.create_optimizer(
            loss, lr, num_train_steps, num_warmup_steps, False)
        # saver = tf.train.Saver()
        # scaf = tf.train.Scaffold(saver=saver)
        tf.Session().run(tf.global_variables_initializer())
        with tf.train.MonitoredTrainingSession(checkpoint_dir=log_dir,
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
                    if step % 100 == 0:
                        print('====> step:{:06d}|{}\t[train loss:{:.3f}]'.format(
                            step, num_train_steps, train_loss))
                        eval_val, train_val = sess._tf_sess().run([predictions, train_predictions], feed_dict)
                        print('question:\t', ''.join(tokenizer.convert_ids_to_tokens(trainDatas['x'][0])))
                        print('groud truth:\t', ''.join(tokenizer.convert_ids_to_tokens(trainDatas['y'][0])))
                        v1 = train_val[0]
                        v1[v1<0] = 100
                        v2 = train_val[1]
                        v2[v2 < 0] = 100
                        v3 = train_val[2]
                        v3[v3 < 0] = 100
                        v4 = train_val[3]
                        v4[v4 < 0] = 100
                        print('train predictions:\t', ''.join(tokenizer.convert_ids_to_tokens(v1)))
                        print('train predictions:\t', ''.join(tokenizer.convert_ids_to_tokens(v2)))
                        print('train predictions:\t', ''.join(tokenizer.convert_ids_to_tokens(v3)))
                        print('train predictions:\t', ''.join(tokenizer.convert_ids_to_tokens(v4)))
                        print(train_val.shape)
                        print('predictions:\t', ''.join(tokenizer.convert_ids_to_tokens(eval_val[0])))
                        print('predictions:\t', ''.join(tokenizer.convert_ids_to_tokens(eval_val[1])))
                        print('predictions:\t', ''.join(tokenizer.convert_ids_to_tokens(eval_val[2])))
                        print('predictions:\t', ''.join(tokenizer.convert_ids_to_tokens(eval_val[3])))
                        print(eval_val.shape)
                        print('\n')
                        eval_log.append(tokenizer.convert_ids_to_tokens(eval_val[0]))
                        eval_log.append(eval_val[0])
                    step += 1
            except KeyboardInterrupt as e:
                # with open('./log/eval_log.txt', 'w', encoding='utf8') as f:
                #     for log in eval_log:
                #         f.write(' '.join(list(log)))
                #         f.write('\n')
                # saver.save(sess._sess, os.path.join(log_dir, 'except_model'), global_step=tf.train.get_or_create_global_step())
                print(e)
            except Exception as e:
                print(e)
if __name__ == '__main__':
    train()










