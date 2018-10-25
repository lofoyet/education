#! /usr/bin/env python
from __future__ import print_function, unicode_literals
import os
import sys
import logging
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lib.rnn_classification_data as sentiment_rnn_data # noqa
from lib.rnn_classification_option import rnn_classification_option_by_default# noqa
from lib.rnn_classification_model import RNNClassificationModel # noqa


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(opts, train, dev):
    tf.reset_default_graph()
    # train&dev generators and embedding matrix:
    rnn_classification_model = RNNClassificationModel(opts)
    rnn_classification_model.build()
    logger.info("done")
    rnn_classification_model.train(train, dev)


if __name__ == '__main__':
    opts = rnn_classification_option_by_default()
    train, dev = sentiment_rnn_data.generate_train_test(
        opts.data_path, opts.nlabels, opts.split_ratio)
    main(opts, train, dev)
