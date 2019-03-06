# !/usr/bin/env python3
import os
import sys
sys.path.append('./bert/')
import bert
from bert import tokenization
import json
import tensorflow as tf
import collections
import fire

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self, x, x_mask, x_seg, x_len, y, y_len):
      self.x = x
      self.x_mask = x_mask
      self.x_seg = x_seg
      self.x_len = x_len
      self.y = y
      self.y_len = y_len

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()


class ChatQADataProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'dialog.txt'), 'train')
    def get_dev_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'dialog.txt'), 'dev')
    def get_test_examples(self, data_dir):
        return self._create_examples(os.path.join(data_dir, 'dialog.txt'), 'test')
    def _create_examples(self, infile, set_type):
        examples = []
        excp_cnt = 0
        norm_cnt = 0
        with open(infile, 'r', encoding='utf8') as f:
            qas_list = json.load(f)
            for i,qa in enumerate(qas_list):
                if 'question' not in qa or 'answer' not in qa:
                    continue
                if len(qa['question']) > 120 or len(qa['answer']) > 100:
                    excp_cnt += 1
                    continue
                if '?' not in qa['question'] and '？' not in qa['question']:
                    continue
                norm_cnt += 1
                guid = "%s-%s" % (set_type, i)
                text_a = tokenization.convert_to_unicode(qa['question'].replace("“",'').replace("”",''))
                text_b = tokenization.convert_to_unicode(qa['answer'].replace("“",'').replace("”",''))
                examples.append(InputExample(guid, text_a, text_b))
        # print('count of exceed 100:\t', excp_cnt)
        # print('count of norm: \t', norm_cnt)
        return examples

def convert_single_example(index, example, max_a_length, max_b_length, tokenizer):
    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = tokenizer.tokenize(example.text_b)
    if len(tokens_a) > max_a_length-2:
        tokens_a = tokens_a[0:max_a_length-2]
    tokens_aa = []
    segment_ids = []
    tokens_aa.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens_aa.append(token)
        segment_ids.append(0)
    tokens_aa.append('[SEP]')
    segment_ids.append(0)
    x = tokenizer.convert_tokens_to_ids(tokens_aa)
    x_seg = segment_ids
    x_mask = [1]*len(x)
    x_len = len(x)
    while len(x) < max_a_length:
        x.append(0)
        x_mask.append(0)
        x_seg.append(0)
    if len(tokens_b) > max_b_length:
        tokens_b = tokens_b[0:max_b_length+1]
    y = tokenizer.convert_tokens_to_ids(tokens_b)
    y_len = len(y)
    while len(y) < max_b_length:
        y.append(0)
    if index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens_aa]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in x]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in x_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in x_seg]))
        tf.logging.info("target_ids: %s" % " ".join([str(x) for x in y]))
        tf.logging.info("x_len: {}".format(x_len))
        tf.logging.info("y_len:{}".format(y_len))
    feature = InputFeatures(x, x_mask, x_seg, x_len, y, y_len)
    return feature

def file_based_convert_examples_to_features(examples, max_a_len, max_b_len, tokenizer, output_file):
    writer = tf.python_io.TFRecordWriter(output_file)
    for index, example in enumerate(examples):
        if index % 10000 == 0:
            print('Writting example {} of {}'.format(index, len(examples)))
        feature = convert_single_example(index, example, max_a_len, max_b_len, tokenizer)
        def create_in_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f
        features = collections.OrderedDict()
        features['x'] = create_in_feature(feature.x)
        features['x_mask'] = create_in_feature(feature.x_mask)
        features['x_seg'] = create_in_feature(feature.x_seg)
        features['y'] = create_in_feature(feature.y)
        features['x_len'] = create_in_feature([feature.x_len])
        features['y_len'] = create_in_feature([feature.y_len])
        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def file_based_input_fn_builder(input_file, max_a_len, max_b_len, is_training, drop_remainer):
    name_to_features = {
        'x': tf.FixedLenFeature([max_a_len], tf.int64),
        'x_mask': tf.FixedLenFeature([max_a_len], tf.int64),
        'x_seg': tf.FixedLenFeature([max_a_len], tf.int64),
        'x_len': tf.FixedLenFeature([], tf.int64),
        'y': tf.FixedLenFeature([max_b_len], tf.int64),
        'y_len': tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in example.keys():
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn(batch_size):
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)
        d = d.apply(tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size = batch_size,
            drop_remainder=drop_remainer
        ))
        return d
    return input_fn


def test_ChatQADataProcessor():
    indir = '../data/dialog'
    processor = ChatQADataProcessor()
    train_examples = processor.get_train_examples(indir)
    max_a_len = 0
    max_b_len = 0
    for example in train_examples:
        if max_a_len < len(example.text_a):
            max_a_len = len(example.text_a)
        if max_b_len < len(example.text_b):
            max_b_len = len(example.text_b)
    for i in range(10):
        print('====> dialog:')
        print('\ta:\t', train_examples[i].text_a)
        print('\tb:\t', train_examples[i].text_b)
        print('\n')
    print('max_a_len:\t', max_a_len)
    print('max_b_len:\t', max_b_len)
    print('dialog count:\t', len(train_examples))


def test_file_based_convert_examples_to_features():
    indir = '../data/dialog'
    train_file = '../data/dialog/train.tfrecord'
    processor = ChatQADataProcessor()
    train_examples = processor.get_train_examples(indir)
    max_a_length = 128
    max_b_length = 100
    vocab_file = './model/chinese_L-12_H-768_A-12/vocab.txt'
    tokenizer = tokenization.FullTokenizer(vocab_file)
    file_based_convert_examples_to_features(train_examples, max_a_length, max_b_length, tokenizer, train_file)

def test_file_based_input_fn_builder():
    input_file = '../data/dialog/train.tfrecord'
    max_a_len = 128
    max_b_len = 100
    input_fn = file_based_input_fn_builder(input_file, max_a_len, max_b_len, True, True)
    batch_size = 32
    ds = input_fn(batch_size)
    iterator = ds.make_one_shot_iterator()
    batch_inputs = iterator.get_next()
    trainDatas = None
    vocab_file = './model/chinese_L-12_H-768_A-12/vocab.txt'
    tokenizer = tokenization.FullTokenizer(vocab_file)
    with tf.Session() as sess:
        trainDatas = sess.run(batch_inputs)
        for i in range(trainDatas['x'].shape[0]):
            a = tokenizer.convert_ids_to_tokens(trainDatas['x'][i])
            b = tokenizer.convert_ids_to_tokens(trainDatas['y'][i])
            print('====> dialog:')
            print('\ta:\t', a)
            print('\tb:\t', b)
            print('\n')
    print('test_file_based_input_fn_builder end!')

if __name__ == '__main__':
    # test_ChatQADataProcessor()
    # test_file_based_convert_examples_to_features()
    test_file_based_input_fn_builder()
    # fire.Fire()

