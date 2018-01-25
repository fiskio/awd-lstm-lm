import itertools as it
import random


class RedditBatchGenerator:

    def __init__(self, reddit_avro_text, n_streams, bptt,
                 max_user_tokens=None, bos=None, eos=None,
                 shuffle_style='classic',
                 # stop iteration if fewer than n_streams users remain
                 strict_n_streams=False,
                 pool_size=None):
        self.usr_msgs = reddit_avro_text.next_user_msgs()
        self.n_streams = n_streams
        self.bptt = bptt
        self.max_user_tokens = max_user_tokens or float('inf')
        self.bos = bos
        self.eos = eos
        self.shuffle_style = shuffle_style
        self.iter_funcs = dict(classic=self.classic_iter,
                               inter_user=self.shuffled_iter)
        assert self.shuffle_style in self.iter_funcs
        self.pool_size = 10 * n_streams if pool_size is None else pool_size
        self.strict_n_streams = strict_n_streams

    def preprocess_msg(self, orig_msg, max_tokens):
        """Takes a vocabulary containing a mapping 'text': <string>
           and adds 3 keys: 'train_chunks', 'label_chunks' and 'start_mask'
        """
        # make copy
        msg = dict(orig_msg)

        # split into tokens
        tokens = orig_msg['text'].split()

        # limit tokens
        if float('inf') > max_tokens > 0:
            tokens = tokens[:max_tokens]

        # add special markers?
        if self.bos is not None:
            tokens = [self.bos] + tokens
        if self.eos is not None:
            tokens += [self.eos]

        # sequences
        train_seq = tokens[:-1]
        label_seq = tokens[1:]

        # position of first item of the chunk in original message
        msg['chunk_offsets'] = range(0, len(label_seq), self.bptt)

        # train
        msg['train_chunks'] = [train_seq[i:i + self.bptt]
                               for i in msg['chunk_offsets']]

        # label
        msg['label_chunks'] = [label_seq[i:i + self.bptt]
                               for i in msg['chunk_offsets']]

        # resets
        msg['start_mask'] = [True if i == 0 else False
                             for i in range(len(train_seq))]
        return msg

    def _msg_filter(self, msgs):
        '''
        A generator yielding only valid messages (currently excludes empty
        ones)
        '''
        for i, m in enumerate(msgs):
            if not len(m['text']):
                print('Skipping zero length message:', i, m)
            else:
                yield m

    def iterate_user_msg_chunks(self, user_msgs):
        # for each sequential message of this user
        n_tokens = 0
        for i, msg in enumerate(self._msg_filter(user_msgs)):

            # maximum number of tokens reached?
            if n_tokens >= self.max_user_tokens:
                break

            # add 'train_chunks', 'label_chunks', 'start_mask'
            remaining_tokens = self.max_user_tokens - n_tokens
            prep_msg = self.preprocess_msg(msg, max_tokens=remaining_tokens)

            # increment user token count
            n_tokens += len(prep_msg['text'])

            # emit one message chunk at the time
            for train_chunk, label_chunks, start_mask, chunk_offset\
                    in zip(prep_msg['train_chunks'],
                           prep_msg['label_chunks'],
                           prep_msg['start_mask'],
                           prep_msg['chunk_offsets']):

                # also add back some useful msg invariant information
                yield dict(time=prep_msg['time'],
                           text=prep_msg['text'],
                           user=prep_msg['user'],
                           topic=prep_msg['topic'],
                           train_chunk=train_chunk,
                           label_chunk=label_chunks,
                           is_first=start_mask,
                           chunk_offset=chunk_offset,
                           msg_id=i)

    def __iter__(self):
        return self.iter_funcs[self.shuffle_style]()

    def classic_iter(self):
        batch = []

        # a pool of n_streams (or less) chunk generators, one per active user
        user_pool = [self.iterate_user_msg_chunks(msgs)
                     for msgs in it.islice(self.usr_msgs, self.n_streams)]

        min_pool_size = self.n_streams if self.strict_n_streams else 0
        while len(user_pool) > min_pool_size:
            curr_user, *rest = user_pool
            user_pool = rest
            try:
                if len(batch) > len(user_pool):
                    yield batch
                    batch = []
                # get next msg chunk for curr_user
                batch.append(next(curr_user))

            except StopIteration:
                # ..unless curr_user is finished
                try:
                    # so fetch a new fresh_user instead
                    next_user_msgs = next(self.usr_msgs)
                    curr_user = self.iterate_user_msg_chunks(next_user_msgs)

                except StopIteration:
                    # ..unless there are mo more users
                    continue
            # finally put curr_user at the end of the queue (round robin)
            user_pool.append(curr_user)

    def queue_next_user(self):
        chunk_gen = None
        try:
            chunk_gen = next(self.usr_msgs)
        except StopIteration:
            pass
        return chunk_gen

    def shuffled_iter(self):
        seed = 42
        rng = random.Random()
        rng.seed(seed)
        # user_pool is a sequence of generators yielding a dict containing
        # a bptt-sized chunk of text
        user_pool = [self.iterate_user_msg_chunks(msgs) for msgs in
                     it.islice(self.usr_msgs, self.pool_size)]

        pool_ids = list(range(len(user_pool)))
        min_pool_size = self.n_streams if self.strict_n_streams else 0
        while len(pool_ids) > min_pool_size:
            batch = []
            # sample n_streams users from pool without replacement
            selected_ids = rng.sample(pool_ids,
                                      min(len(pool_ids), self.n_streams))
            for idx in selected_ids:
                msg_chunk = None
                try:
                    msg_chunk = next(user_pool[idx])
                except StopIteration:
                    keep_queuing = True
                    while keep_queuing:
                        # try to queue the next user
                        chunk_gen = self.queue_next_user()
                        if not chunk_gen:
                            # no more users, remove that lane from the pool
                            # and continue with any others that remain
                            pool_ids.remove(idx)
                            break
                        # user loaded, try to read a chunk from them
                        user_pool[idx] = self.iterate_user_msg_chunks(chunk_gen)  # noqa
                        try:
                            # try to read first chunk from new user
                            # catch in case there is no data (possibly through
                            # filtering)
                            msg_chunk = next(user_pool[idx])
                            break
                        except StopIteration:
                            # stop if users have alreaady been exhausted
                            if not chunk_gen:
                                pool_ids.remove(idx)
                                break
                if msg_chunk is not None:
                    batch.append(msg_chunk)
            yield batch


def cycle(iterable):
    """Streaming replacement for itertools.cycle.

    itertools.cycle is designed to work with one-shot iterators, so
    brings the entire iterator into memory in order to repeat it. This
    handy feature is known in contexts such as ours as a 'huge memory
    leak'.

    This requires its argument to be a (multi-shot) iterable,
    supporting multiple calls to __iter__.

    """
    while True:
        yield from iterable


class NebulaBatchGenerator:
    """Generates batches of 'sentence tokens' from a Nebula text corpus.
    It reads 'max_lines_per_epoch' lines at the time, groups them by number
    of words into batches of at most 'batch_size'. Batch size is the product
    of 'sentence length' and 'number of sentences'. Each item is an epoch, a
    list of dictionaries of features. Each batch is a dictionary of lists, with
    at least the key 'text'. Each key in a batch is associated with a list of
    'sentence length'
    """

    def __init__(self, corpus, max_lines_per_epoch=None, max_epochs=None,
                 batch_size=256, min_sent_len=2, max_sent_len=20,
                 max_reps=5, cycle=False,
                 tokenizer=str.split, bos=None, eos=None):
        """`corpus`: iterable of dicts with at least a 'text' key, usually
        from 'corpus_utils'

        `max_lines_per_epoch`: where specified, each epoch is based on
        reading a chunk with at most this number of lines from the
        corpus. This is useful for large corpora and online learning
        settings. When not specified, an epoch is a full pass over the
        `corpus`, which is the more traditional definition of an epoch
        used when working with smaller datasets or batch learning.

        `cycle`: when the corpus is exhausted, cycle around again back
        to the start.

        `max_epochs`: if specified, limit to maximum number of epochs

        `batch_size`: upper bound for the product of 'sentence length'
                      and 'number of sentences'

        `max_sent_len`: truncate lines longer than this

        `min_sent_len`: skip lines shorter than this

        `max_reps`: maximum allowed word repetitions in a sentence

        `tokenizer`: tokenizer function to use. Defaults to
        `str.split`, for pre-tokenized data that was joined by
        whitespace.

        Returns a generator of batches form a corpus

        """
        self.corpus = corpus
        self.max_lines_per_epoch = max_lines_per_epoch
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.max_sent_len = max_sent_len
        self.min_sent_len = min_sent_len
        self.max_reps = max_reps
        self.cycle = cycle
        self.tokenizer = tokenizer
        self.bos = bos
        self.eos = eos
        self.msg_id = 0

    def __iter__(self):
        """"
        return a iterator of one batch only
        """
        for epoch in self.epochs():
            yield from self.batch_epoch(epoch)

    def epochs(self):
        """A generator of epochs at most 'max_lines_per_epoch' lines from a
        corpus, or of full passes over the corpus

        """
        if self.max_lines_per_epoch is None:
            epochs = cycle([self.corpus]) if self.cycle else [self.corpus]

        else:
            corpus_iter = cycle(self.corpus) if self.cycle else \
                iter(self.corpus)

            def gen_epochs():
                while True:
                    epoch_iter = it.islice(
                        corpus_iter, self.max_lines_per_epoch)
                    # We need to peek ahead to find out if there are any
                    # remaining items before yielding a new epoch. If
                    # there aren't, the call to next will raise
                    # StopIteration which will end this generator before
                    # it yields
                    yield it.chain([next(epoch_iter)], epoch_iter)
            epochs = gen_epochs()

        if self.max_epochs:
            epochs = it.islice(epochs, self.max_epochs)

        return epochs

    def batch_epoch(self, epoch):
        """
        :param epoch: a list of lines from a corpus
                      (aka: dictionaries with at least the key "text")
        :return: a list (batches) of dictionaries (features)
                 all batches have at most "batch_size" items in each list
                 a batch has at most 'sentence length' x 'num sentences' items
                 with added keys: 'train_chunk', 'label_chunk' and 'is_first'
        """
        line_len_to_inprogress_batch = {}
        all_batches = []

        for i, item in enumerate(epoch):
            tokens = self.tokenizer(item["text"])
            # some null bytes actually occurred in tokens;
            # round-tripping these via numpy string arrays ends up
            # with empty string when at start of the word, as they're
            # null-padded/terminated. Best to drop them!
            tokens = [t for t in tokens if t[0] != '\x00']
            tokens = tokens[:self.max_sent_len]

            # check for excessive repetition
            reps = [sum(1 for _ in group)
                    for key, group in it.groupby(tokens)]
            if len(reps) > 0 and max(reps) > self.max_reps:
                continue

            line_len = len(tokens)

            # check for lines that are too short
            if self.min_sent_len is not None and line_len < self.min_sent_len:
                continue

            # add special markers?
            if self.bos is not None:
                tokens = [self.bos] + tokens
            if self.eos is not None:
                tokens += [self.eos]

            # add train_chunk and label_chunk

            item['train_chunk'] = tokens[:-1]
            item['label_chunk'] = tokens[1:]
            # always true for now as we do not break text into chunks
            item['is_first'] = True
            item['chunk_offset'] = 0
            item['msg_id'] = self.msg_id
            # users must be unique in each batch so just number them
            item['user'] = i
            self.msg_id += 1

            if line_len not in line_len_to_inprogress_batch:
                line_len_to_inprogress_batch[line_len] = []
            line_len_to_inprogress_batch[line_len].append(item)
            batch_len = len(line_len_to_inprogress_batch[line_len])
            # if the current batch is now full then append to all_batches
            # ready to be returned and create a new batch of that line_len
            if line_len * batch_len >= self.batch_size:
                all_batches.append(line_len_to_inprogress_batch[line_len])
                line_len_to_inprogress_batch[line_len] = []

        # some batches left over which aren't full yet. for now we're
        # choosing to use these, even though they can potentially be
        # much smaller than the target batch size.
        all_batches.extend(filter(lambda b: 0 < len(b),
                                  line_len_to_inprogress_batch.values()))
        return all_batches

