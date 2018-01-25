import itertools as it

from corpus_utils.vocab import RedditVocab, NebulaVocab


class Vocab:
    """A mapping between TERMs and IDs

       Additionally might add special tokens for:
         - begin-of-sequence
         - end-of-sequence
         - out-of-vocabulary
    """

    def __init__(self, init_terms, oov=None, bos=None, eos=None, name=None):
        self.init_terms = init_terms if init_terms is not None else []
        self.init_size = len(self.init_terms)
        self.name = name

        # special tokens
        self.specials = []
        if oov is not None:
            self.specials.append(oov)
        if bos is not None:
            self.specials.append(bos)
        if eos is not None:
            self.specials.append(eos)

        # add init tokens
        self._index_to_term = list(it.chain(self.specials, self.init_terms))
        self._term_to_index = {t: i for i, t in enumerate(self._index_to_term)}
        self.oov_idx = None
        self.oov = oov
        self.oov_idx = self.get_index(oov, grow_if_missing=False) \
            if oov is not None else None

        self.bos = bos
        self.bos_idx = self.get_index(bos, grow_if_missing=False) \
            if bos is not None else None

        self.eos = eos
        self.eos_idx = self.get_index(eos, grow_if_missing=False) \
            if eos is not None else None

    def get_index(self, term, grow_if_missing=False):
        """Return the index of a given term
           `grow_if_missing` defines the behaviour for new terms
              - True:  term is added to user_vocab, new index is returned
              - False: term is NOT added to user_vocab, `oov_index` is returned
        """
        if grow_if_missing and term not in self._term_to_index:
            self._index_to_term.append(term)
            self._term_to_index[term] = len(self._index_to_term) - 1
        return self._term_to_index.get(term, self.oov_idx)

    def get_term(self, index):
        """Return the term given an index"""
        return self._index_to_term[index]

    def __eq__(self, other):
        return isinstance(other, Vocab) and \
               self._index_to_term == other._index_to_term

    def __len__(self):
        return len(self._index_to_term)

    def __repr__(self):
        return 'Vocab(%s)' % self.name

    def __iter__(self):
        return iter(self._index_to_term)

    def __contains__(self, key):
        return key in self._index_to_term

    def __hash__(self):
        return hash(tuple(self._index_to_term))

    @classmethod
    def empty(cls):
        return cls(init_terms=[])

    @classmethod
    def from_unigram_file(cls, unig_file,
                          max_len=None, oov=None, bos=None, eos=None):
        """Factory to initialise Vocab from unigram file"""
        with open(unig_file) as vf:
            init_words = [line.split()[0] for line in it.islice(vf, max_len)]
            return cls(init_words, oov=oov, bos=bos, eos=eos)

    @classmethod
    def reddit(cls, vocab_size=9998, oov='_OOV_', bos='_BOS_', eos=None):
        """Factory to initialise a standard Reddit vocabulary"""
        init_words = [term['text'] for term in
                      RedditVocab(vocab_size=vocab_size)]
        return cls(init_words, oov=oov, bos=bos, eos=eos)

    @classmethod
    def nebula(cls, locale, vocab_size=9998,
               oov='_OOV_', bos='_BOS_', eos=None):
        init_words = [term['text'] for term in
                      NebulaVocab(locale=locale, vocab_size=vocab_size)]
        return cls(init_words, oov=oov, bos=bos, eos=eos)

    def init_args(self):
        return dict(init_terms=self._index_to_term[len(self.specials):],
                    bos=self.bos,
                    eos=self.eos,
                    oov=self.oov,
                    name=self.name)

    @classmethod
    def clone(cls, orig_vocab, name=None):
        init_args = orig_vocab.init_args()
        if name is not None:
            init_args['name'] = name
        return cls(**init_args)

    def subtract(self, other_vocab):
        """Return the delta terms with another vocabulary"""
        assert self._index_to_term[:len(other_vocab)] == list(other_vocab)
        return self._index_to_term[len(other_vocab):]


class UserVocab:
    """
    Combines an immutable starting vocab with a working user vocab, removing
    the need to copy the starting vocab's contents when initialising a fresh
    vocab for a new user.

    Due to the more complicated accesses across the two vocabs, this makes term
    index lookups more expensive compared to a plain dict[] access of a plain
    Vocaab's _term_index dict but trades this for much faster initialisation.
    """

    def __init__(self, base_vocab, name=None):
        '''
        base_vocab - The Vocab object stroing all currently known terms.
                     New terms will be added to a separate Vocab object.
        name - The name of this object (for __repr__)
        '''
        self.base_vocab = base_vocab
        self.init_size = len(base_vocab)
        self.name = name
        self.oov = base_vocab.oov
        self.oov_idx = base_vocab.oov_idx
        self.bos = base_vocab.bos
        self.bos_idx = base_vocab.bos_idx
        self.eos = base_vocab.eos
        self.eos_idx = base_vocab.eos_idx
        self.user_vocab = Vocab.empty()

    def get_index(self, term, grow_if_missing=False):
        """Return the index of a given term
           `grow_if_missing` defines the behaviour for new terms
              - True:  term is added to user_vocab, new index is returned
              - False: term is NOT added to user_vocab, `oov_index` is returned
        """
        if term in self.base_vocab:
            return self.base_vocab.get_index(term, grow_if_missing=False)
        if term in self.user_vocab:
            return self.init_size + \
                   self.user_vocab.get_index(term, grow_if_missing=False)
        if grow_if_missing:
            return self.user_vocab.get_index(term, grow_if_missing)
        return self.base_vocab.oov

    def get_term(self, index):
        """Return the term given an index"""
        if index < self.init_size:
            return self.base_vocab.get_term(index)
        else:
            return self.user_vocab.get_term(index - self.init_size)

    def __eq__(self, other):
        return isinstance(other, UserVocab) and \
               self.base_vocab == other.base_vocab and \
               self.user_vocab == other.user_vocab

    def __len__(self):
        return self.init_size + len(self.user_vocab)

    def __repr__(self):
        return 'UserVocab(%s: %s + %s)' % (self.name,
                                           self.base_vocab.__repr__(),
                                           self.user_vocab.__repr__())

    def __iter__(self):
        return it.chain(iter(self.base_vocab), iter(self.user_vocab))

    def __contains__(self, key):
        return key in self.base_vocab or key in self.user_vocab


class VocabCache:
    """A cache of user vocabularies

       Its purpose is to keep track of the working vocabulary of each user.
       Given a batch it augments every message chunk in it with a new key
       'user_vocab' that is the current vocabulary of that user.

       If a user does not have a vocabulary it gets a copy of the base_vocab,
       which is either empty or a Vocab given to the constructor of VocabCache.
    """

    def __init__(self, base_vocab=None):
        self.base_vocab = Vocab.clone(base_vocab)
        self.cache = {}
        self.stash = {}

    def get_user_vocab(self, user):
        if user not in self.cache:
            self.cache[user] = Vocab.clone(self.base_vocab, name=user)
            self.stash[user] = list()
        return self.cache[user]

    def fetch_vocabs(self, batch):
        """Add key 'user_vocab' to all messages in a batch
           Create one by copying `base_vocab` for new users
        """
        for msg in batch:
            user = msg['user']
            user_vocab = self.get_user_vocab(user)

            if msg['is_first']:
                # learn all stashed vocab for this user
                for term in self.stash[user]:
                    user_vocab.get_index(term, grow_if_missing=True)
                self.stash[user] = list()

            # stash all message terms to be learnt later
            terms = set()
            for term in msg['label_chunk']:
                if term not in terms:
                    self.stash[user].append(term)
                    terms.add(term)

            # add user vocab to the current message in batch
            msg['user_vocab'] = user_vocab
            msg['user_vocab_size'] = len(user_vocab)
        return batch

    def __len__(self):
        return len(self.cache)

    def __repr__(self):
        return 'VocabCache(%d)' % len(self)

    def __iter__(self):
        return iter(self.cache.values())

    def __contains__(self, key):
        return key in self.cache
