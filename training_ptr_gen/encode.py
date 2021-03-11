from __future__ import unicode_literals, print_function, division

import os
import time
import argparse

import tensorflow as tf
import torch
from model import Model
from torch.nn.utils import clip_grad_norm_

from torch.optim import Adagrad

from data_util import config
from data_util.batcher import Batcher
from data_util.data import Vocab
from data_util.utils import calc_running_avg_loss
from train_util import get_input_from_batch, get_output_from_batch

use_cuda = config.use_gpu and torch.cuda.is_available()

class Encode(object):
    def __init__(self, model_file_path, destination_dir):
        self.vocab = Vocab(config.vocab_path, config.vocab_size)
        self.batcher = Batcher(config.encode_data_path, self.vocab, mode='encode',
                               batch_size=config.batch_size, single_pass=True)
        time.sleep(5)

        self.output = {}
        self.destination_dir = destination_dir
        self.model = Model(model_file_path, is_eval=True)

    def save_output(self, output, destination_dir):
        if destination_dir is None:
            torch.save(output, "output")
        else:
            torch.save(output, destination_dir)

    def encode_one_batch(self, batch):
        enc_batch, enc_padding_mask, enc_lens, enc_batch_extend_vocab, extra_zeros, c_t_1, coverage = \
            get_input_from_batch(batch, use_cuda)

        encoder_outputs, encoder_feature, encoder_hidden = self.model.encoder(enc_batch, enc_lens)

        h, c = self.model.reduce_state(encoder_hidden)
        h, c = h.squeeze(0), c.squeeze(0)
        encodes = torch.cat((h, c), 1)

        for id, encode in zip(batch.original_abstracts, encodes):
            print(encode)
            self.output[id] = encode

    def run_encode(self):
        start = time.time()
        batch = self.batcher.next_batch()
        while batch is not None:
            self.encode_one_batch(batch)
            batch = self.batcher.next_batch()
        self.save_output(self.output, self.destination_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Encode script")
    parser.add_argument("-m",
                        dest="model_file_path", 
                        required=False,
                        default=None,
                        help="Model file for encoding (default: None).")
    parser.add_argument("-d",
                        dest="destination_dir", 
                        required=False,
                        default=None,
                        help="Destination folder for encoding (default: None).")
    args = parser.parse_args()
    
    encoder = Encode(args.model_file_path, args.destination_dir)
    encoder.run_encode()