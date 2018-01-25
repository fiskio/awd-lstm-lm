#!/usr/bin/env python3
import argparse
import pprint
import time
import math
import os
import torch

from torch.autograd import Variable
from datetime import datetime

from corpus_utils import RedditAvroText, NebulaText, NebulaTestSet
from yazz.data import NebulaBatchGenerator, RedditBatchGenerator
from yazz.vocabo import Vocab
from yazz.modelo import RNNModel


def get_name():
    return '_'.join(str(datetime.now()).split('.')[0].split())


class Trainer:

    def preprocess_msg(self, msg, max_user_tokens):
        tokens = msg['text'].split()
        if self.num_tokens + len(tokens) > max_user_tokens:
            tokens = tokens[:max_user_tokens - self.num_tokens]
        self.num_tokens += len(tokens)
        tokens = ['_BOS_'] + tokens
        assert len(tokens) >= 2
        msg['train_chunk'] = tokens[:-1]
        msg['label_chunk'] = tokens[1:]
        msg['is_first'] = True
        return msg

    def prepare_reddit(self, train_users,
                       valid_users,
                       n_streams,
                       bptt,
                       max_user_tokens,
                       shuffle_style,
                       pool_size,
                       strict_n_streams):

        self.vocab = Vocab.reddit()
        train_data = RedditAvroText(max_users=train_users, dataset='train')
        self.train_bg = iter(RedditBatchGenerator(
                             train_data,
                             n_streams=n_streams,
                             bptt=bptt,
                             max_user_tokens=max_user_tokens,
                             bos=self.vocab.bos,
                             shuffle_style=shuffle_style,
                             pool_size=pool_size,
                             strict_n_streams=strict_n_streams))

        valid_data = RedditAvroText(max_users=valid_users, dataset='valid')
        valid_user_files = valid_data.get_user_files()
        valid_msgs = []

        for valid_user_file in valid_user_files:
            valid_user_msgs = RedditAvroText.get_user_msgs(valid_user_file)
            self.num_tokens = 0

            for msg in valid_user_msgs:
                msg = self.preprocess_msg(msg, max_user_tokens)
                valid_msgs.append(msg)

                if self.num_tokens >= max_user_tokens:
                    break

        self.valid_bg = []
        for i in range(0, len(valid_msgs), n_streams):
            batch = valid_msgs[i:i + n_streams]
            self.valid_bg.append(batch)

    def prepare_nebula(self, locale,
                       vocab_size, batch_size,
                       max_epochs, max_lines_per_epoch,
                       cycle):

        self.vocab = Vocab.nebula(locale, vocab_size)
        train_nbt = NebulaText(locale=locale)
        self.train_bg = iter(NebulaBatchGenerator(
                             corpus=train_nbt, batch_size=batch_size,
                             max_lines_per_epoch=max_lines_per_epoch,
                             max_epochs=max_epochs, bos=self.vocab.bos,
                             cycle=cycle))

        valid_nbt = NebulaTestSet(locale=locale)
        valid_batch_size = max(128, batch_size)
        valid_max_lines_per_epoch = max(1000, max_lines_per_epoch)
        valid_max_epochs = max(10, max_epochs)
        self.valid_bg = list(NebulaBatchGenerator(
                             corpus=valid_nbt, batch_size=valid_batch_size,
                             max_lines_per_epoch=valid_max_lines_per_epoch,
                             max_epochs=valid_max_epochs,
                             cycle=cycle))


    def batchify(self, batch):
        train_b = [[self.vocab.get_index(word) for word in line['train_chunk']]
                   for line in batch]
        train_b = Variable(torch.LongTensor(train_b))

        label_b = [[self.vocab.get_index(word) for word in line['label_chunk']]
                   for line in batch]
        label_b = Variable(torch.LongTensor(label_b).view(-1))
        return train_b, label_b

    def repackage_hidden(self, h):
        """Wraps hidden states in new Variables, to detach them from their history."""
        if type(h) == Variable:
            return Variable(h.data)
        else:
            return tuple(self.repackage_hidden(v) for v in h)

    def train(self, batch, i):

        # preprocess batch: words -> indices
        train_batch, target_batch = self.batchify(batch)

        # training time
        if args.rnn_type == 'QRNN': self.model.reset()
        self.model.train()

        start_time = time.time()
        hidden = self.model.init_hidden(train_batch.size(1))

        # adjust learning rate according to batch size
        # TODO: this seq_len vs bptt is weird...
        batch_size = train_batch.size(0)
        seq_len = train_batch.size(1)
        # TODO
        lr2 = self.optimizer.param_groups[0]['lr']
        #self.optimizer.param_groups[0]['lr'] = lr2 * batch_size / seq_len

        self.optimizer.zero_grad()

        output, hidden = self.model(train_batch, hidden)

        flat_output = output.view(-1, len(self.vocab))
        raw_loss = self.criterion(flat_output, target_batch)

        # TODO
        # Activiation Regularization
        # loss = loss + sum(
        #     args.alpha * dropped_rnn_h.pow(2).mean() for dropped_rnn_h
        #     in dropped_rnn_hs[-1:])

        # TODO
        # Temporal Activation Regularization (slowness)
        # loss = loss + sum(
        #     args.beta * (rnn_h[1:] - rnn_h[:-1]).pow(2).mean() for
        #     rnn_h in rnn_hs[-1:])
        raw_loss.backward()

        torch.nn.utils.clip_grad_norm(self.model.parameters(), args.clip)
        self.optimizer.step()

        self.optimizer.param_groups[0]['lr'] = lr2
        if i % args.log_interval == 0:
            elapsed = time.time() - start_time
            cur_loss = raw_loss.data[0]
            print(
                '| {:5d} batch | lr {:02.2f} | ms/batch {:5.2f} | '
                'loss {:5.2f} | ppl {:8.2f}'.format(
                    i,
                    self.optimizer.param_groups[0]['lr'],
                    elapsed * 1000 / args.log_interval,
                    cur_loss,
                    math.exp(cur_loss)))

    def loop(self, args):
        # model
        self.model = RNNModel(rnn_type=args.rnn_type,
                              ntoken=len(self.vocab),
                              ninp=args.emb_size,
                              nhid=args.rnn_size,
                              nlayers=args.n_layers,
                              dropout=args.dropout,
                              dropouti=args.dropouti,
                              dropoute=args.dropoute,
                              dropouth=args.dropouth,
                              wdrop=args.wdrop,
                              tie_weights=True,
                              byte=False)

        # optimizer
        train_params = filter(lambda p: p.requires_grad, self.model.parameters())

        # TODO: try Adam
        self.optimizer = torch.optim.SGD(train_params,
                                         lr=args.learning_rate,
                                         weight_decay=args.wdecay)

        self.criterion = torch.nn.CrossEntropyLoss()

        # 🚂 🚂 🚂
        for i, batch in enumerate(self.train_bg):

            # train
            self.train(batch, i)

            if i % args.valid_interval == 0 and i > 0:

                # validation
                valid_err = []
                self.model.eval()
                if args.rnn_type == 'QRNN': self.model.reset()

                for vbatch in self.valid_bg:
                    # preprocess batch: words -> indices
                    valid_batch, target_batch = self.batchify(vbatch)
                    vhidden = self.model.init_hidden(valid_batch.size(1))
                    output, hidden = self.model(valid_batch, vhidden)
                    flat_output = output.view(-1, len(self.vocab))
                    raw_loss = self.criterion(flat_output, target_batch)
                    valid_err.append(raw_loss[0])

                v_avg_loss = sum(valid_err) / len(valid_err)
                # save
                checkpoint = 'model_%.2f_%i.cntk' % (v_avg_loss, i)
                checkpoint = os.path.join('models', args.model_name, checkpoint)
                with open(checkpoint, 'wb') as f:
                    torch.save(self.model, f)
                print('Model saved at %s' % checkpoint)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default=get_name(),
                        help='output model folder name')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for repeatability')
    parser.add_argument('--emb-size', type=int, default=160,
                        help='word embedding size')
    parser.add_argument('--rnn-size', type=int, default=512,
                        help='RNN hidden state size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='training log interval')
    parser.add_argument('--valid-interval', type=int, default=5000,
                        help='validation log interval')
    parser.add_argument('--rnn-type', type=str,
                        choices=('LSTM', 'QRNN', 'GRU'),
                        default='GRU',
                        help='Type of RNN to use')
    parser.add_argument('--vocab-size', type=int, default=10000,
                        help='vocabulary size limit')
    parser.add_argument('--n-layers', type=int, default=1,
                        help='number of hidden layers')

    parser.add_argument('--dropout', type=float, default=0,
                        help='amount of dropout')
    parser.add_argument('--dropouti', type=float, default=0,
                        help='amount of dropout I')
    parser.add_argument('--dropoute', type=float, default=0,
                        help='amount of dropout E')
    parser.add_argument('--dropouth', type=float, default=0,
                        help='amount of dropout H')
    parser.add_argument('--wdrop', type=float, default=0,
                        help='amount of weight drop')
    parser.add_argument('--wdecay', type=float, default=0,
                        help='amount of weight decay')
    parser.add_argument('--alpha', type=float, default=0,
                        help='alpha L2 regularization on RNN activation (alpha = 0 means no regularization)')
    parser.add_argument('--beta', type=float, default=0,
                        help='beta slowness regularization applied on RNN activiation (beta = 0 means no regularization)')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')

    subparsers = parser.add_subparsers(dest='corpus',
                                       help='available corpus')
    subparsers.required = True

    # train with reddit data
    parser_reddit = subparsers.add_parser('reddit',
                                          help='train with reddit data')
    parser_reddit.add_argument('--train-users', type=int, default=1000000,
                               help='num of users for training')
    parser_reddit.add_argument('--valid-users', type=int, default=1000,
                               help='num of users for validation')
    parser_reddit.add_argument('--n-streams', type=int, default=64,
                               help='number of users to train in parallel')
    parser_reddit.add_argument('--max-user-tokens', type=int, default=10000,
                               help='max number of training tokens per user')
    parser_reddit.add_argument('--bptt', type=int, default=25,
                               help='back-propagate-through-time length')
    parser_reddit.add_argument('--data-shuffle-style', type=str,
                               choices=('classic', 'inter_user'),
                               default='inter_user',
                               help='Method of shuffling user messages; classic - no shuffling, inter_user - select users for each batch at random from a larger pool of current users')  # noqa
    parser_reddit.add_argument('--pool-size', type=int, default=None,
                               help='For inter_user shuffling, the number of users to select each batch from (None to use 10 x n_streams)')  # noqa
    parser_reddit.add_argument('--strict-n-streams', default=True,
                               action='store_true',
                               help='Stop training when number of remaining users is below n_streams?')  # noqa

    # train with nebula data
    parser_nebula = subparsers.add_parser('nebula',
                                          help='train with nebula data')
    parser_nebula.add_argument('--locale', type=str, default='en_US',
                               help='language locale')
    parser_nebula.add_argument('--batch-size', type=int, default=5120,
                               help='batch size')
    parser_nebula.add_argument('--max-epochs', type=int, default=600,
                               help='max no. of epochs')
    parser_nebula.add_argument('--max-lines-per-epochs',
                               type=int, default=1000000,
                               help='maximum lines used per epoch')
    parser_nebula.add_argument('--cycle', default=False, action='store_true',
                               help='cycle around the data when exhausted?')

    args = parser.parse_args()
    pprint.pprint(vars(args))

    lm_trainer = Trainer()
    if args.corpus == 'reddit':
        lm_trainer.prepare_reddit(args.train_users,
                                 args.valid_users,
                                 args.n_streams,
                                 args.bptt,
                                 args.max_user_tokens,
                                 args.data_shuffle_style,
                                 args.pool_size,
                                 args.strict_n_streams)
    elif args.corpus == 'nebula':
        lm_trainer.prepare_nebula(args.locale,
                                 args.vocab_size,
                                 args.batch_size,
                                 args.max_epochs,
                                 args.max_lines_per_epochs,
                                 args.cycle)
    # just do it!
    lm_trainer.loop(args)
