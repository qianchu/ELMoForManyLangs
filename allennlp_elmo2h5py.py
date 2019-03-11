from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = 'qianchu_liu'

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Extract pre-computed feature vectors from a PyTorch BERT model."""



import argparse
import collections
import logging
import json
import re

import torch
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler


import numpy
from allennlp.modules.elmo import Elmo, batch_to_ids

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
import h5py


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, orig_to_tok_maps, orig_tokens):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        self.orig_to_tok_maps = orig_to_tok_maps
        self.orig_tokens = orig_tokens


def tokenize_map(orig_tokens, tokenizer):
    ### Input
    labels = ["NNP", "NNP", "POS", "NN"]

    ### Output
    bert_tokens = []

    # Token map will be an int -> int mapping between the `orig_tokens` index and
    # the `bert_tokens` index.
    orig_to_tok_map = []

    for orig_token in orig_tokens:
        orig_to_tok_map.append(len(bert_tokens) + 1)
        bert_tokens.extend(tokenizer.tokenize(orig_token))
    return bert_tokens, orig_to_tok_map


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        orig_tokens = example.text_a.split()
        tokens_a, orig_to_tok_map_a = tokenize_map(orig_tokens, tokenizer)
        tokens_b = None
        if example.text_b:
            tokens_b, orig_to_tok_map_b = tokenize_map(example.text_b.split(), tokenizer)
            orig_tokens += example.text_b.split()
            orig_to_tok_map_a += orig_to_tok_map_b
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                print('exceed length:', tokens_a)
                continue  # skip when the length exceeds
                # tokens_a = tokens_a[0:(seq_length - 2)]

        # orig_to_tok_maps.append(orig_to_tok_map_a)
        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0      0   0    1  1  1   1  1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        while len(orig_to_tok_map_a) < seq_length:
            orig_to_tok_map_a.append(0)

        assert len(orig_to_tok_map_a) == seq_length
        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (example.unique_id))
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "input_type_ids: %s" % " ".join([str(x) for x in input_type_ids]))

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                orig_to_tok_maps=orig_to_tok_map_a,
                orig_tokens=orig_tokens))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_examples(input_file, example_batch,max_length):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    with open(input_file, "r", encoding='utf-8') as reader:
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip().split('\t')[0]
            line_lst=line.split()
            if len(line_lst)>max_length:
                continue

            examples.append(
                line_lst)
            if len(examples) >= example_batch:
                yield examples
                examples = []
    if examples != []:
        yield examples


def get_orig_seq(input_mask_batch):
    seq = [i for i in input_mask_batch if i != 0]
    return seq


def feature_orig_to_tok_map(average_layer_batch, orig_to_token_map_batch, input_mask_batch):
    average_layer_batch_out = []
    for sent_i, sent_embed in enumerate(average_layer_batch):
        sent_embed_out = []
        orig_to_token_map_batch_sent = get_orig_seq(orig_to_token_map_batch[sent_i])
        seq_len = len(get_orig_seq(input_mask_batch[sent_i]))

        for i in range(len(orig_to_token_map_batch_sent)):
            start = orig_to_token_map_batch_sent[i]
            if i == (len(orig_to_token_map_batch_sent) - 1):
                sent_embed_out.append(sum(sent_embed[start:seq_len - 1]) / (seq_len - 1 - start))
                continue
            end = orig_to_token_map_batch_sent[i + 1]
            sent_embed_out.append(sum(sent_embed[start:end]) / (end - start))
        average_layer_batch_out.append(numpy.array(sent_embed_out))
    return numpy.array(average_layer_batch_out)

def data_loader(examples_batch,batch_size):
    examples=[]
    for example in examples_batch:
        examples.append(example)
        if len(examples)>=batch_size:
            yield examples
            examples=[]
    if examples!=[]:
        yield examples


def examples2embeds(examples_batch, model,device, writer, args):
    # features = convert_examples_to_features(
    #     examples=examples, seq_length=args.max_seq_length, tokenizer=tokenizer)
    #
    # unique_id_to_feature = {}
    # for feature in features:
    #     unique_id_to_feature[feature.unique_id] = feature
    #
    # all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    # all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    # all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    # all_input_orig_to_token_maps = torch.tensor([f.orig_to_tok_maps for f in features], dtype=torch.long)
    #
    # eval_data = TensorDataset(all_input_ids, all_input_mask, all_example_index, all_input_orig_to_token_maps)
    # if args.local_rank == -1:
    #     eval_sampler = SequentialSampler(examples)
    # else:
    #     eval_sampler = DistributedSampler(examples)
    eval_dataloader = data_loader(examples_batch,  batch_size=args.batch_size)
    model.to(device)
    model.eval()
    batch_counter = 0
    # sent_set = set()
    # with h5py.File(args.output_file, 'w') as writer:
    for examples in eval_dataloader:
        print('batch no. {0}'.format(batch_counter))
        batch_counter += 1

        character_ids = batch_to_ids(examples).to(device)

        embeddings = model(character_ids)
        average_layer_batch=embeddings['elmo_representations'][0].detach().cpu().numpy()
        # all_encoder_layers = embeddings

        # average_layer_batch = sum(all_encoder_layers) / len(all_encoder_layers[0])
        # if orig_to_token_map_batch!=None:


        for b, example in enumerate(examples):

            sent = '\t'.join(example)
            sent = sent.replace('.', '$period$')
            sent = sent.replace('/', '$backslash$')
            # if sent in sent_set:
            #     continue
            # sent_set.add(sent)
            if sent not in writer:
                payload = average_layer_batch[b][:len(example)]

                try:
                    writer.create_dataset(sent, payload.shape, dtype='float32', compression="gzip", compression_opts=9,
                                          data=payload)
                except OSError as e:
                    print(e, sent)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--input_file", default=None, type=str, required=True)
    parser.add_argument("--output_file", default=None, type=str, required=True)

    ## Other parameters
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    # parser.add_argument("--layers", default="-1,-2,-3,-4", type=str)

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for predictions.")
    parser.add_argument('--example_batch', default=100000, type=int, help='batch size for input examples')
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gpu', type=int, help='specify the gpu to use')
    parser.add_argument('--sent_max', type=str, help='sent maximum lenght')

    args = parser.parse_args()

    writer = h5py.File(args.output_file, 'w')

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:{0}".format(args.gpu) if torch.cuda.is_available() and not args.no_cuda else "cpu")
        # n_gpu = torch.cuda.device_count()
        n_gpu = 1
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {} distributed training: {}".format(device, n_gpu, bool(args.local_rank != -1)))

    # layer_indexes = [int(x) for x in args.layers.split(",")]


    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    # Compute two different representation for each token.
    # Each representation is a linear weighted combination for the
    # 3 layers in ELMo (i.e., charcnn, the outputs of the two BiLSTM))
    model = Elmo(options_file, weight_file, 1, dropout=0)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    # elif n_gpu > 1:
    #    model = torch.nn.DataParallel(model)

    example_counter = 0
    for examples in read_examples(args.input_file, args.example_batch,args.sent_max):
        example_counter += 1
        print('processed {0} examples'.format(str(args.example_batch * example_counter)))
        examples2embeds(examples, model, device, writer, args)
    writer.close()


if __name__ == "__main__":
    main()
