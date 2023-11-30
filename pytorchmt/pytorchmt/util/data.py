
import time
import pickle

import torch
from numpy import dtype
import horovod.torch as hvd
from numpy.random import RandomState

from pytorchmt.util.debug import my_print
from pytorchmt.util.globals import Globals
from pytorchmt.util.threading import Thread, SharedMemory


class Vocabulary:

    UNK = "<UNK>"
    PAD = "<PAD>"
    EOS = "</S>"

    special_tokens = [
        UNK,
        PAD,
        EOS
    ]

    def __init__(self, vocab, vocab_size, vocab_rev, vocab_path):
        
        self.vocab = vocab
        self.vocab_size = vocab_size
        self.vocab_rev = vocab_rev
        self.vocab_path = vocab_path

        self.PAD = self.vocab[Vocabulary.PAD]
        self.UNK = self.vocab[Vocabulary.UNK]
        self.EOS = self.vocab[Vocabulary.EOS]

    @staticmethod
    def create_vocab(vocab_path):

        vocab = Vocabulary.read_from_pickle(vocab_path)
        vocab = Vocabulary.append_special_tokens(vocab)
        vocab = Vocabulary.remove_sos_symbol_from_vocabs(vocab)

        vocab_size  = len(vocab.items())
        vocab_rev   = {y:x for x,y in vocab.items()}

        return Vocabulary(vocab, vocab_size, vocab_rev, vocab_path)

    @staticmethod
    def read_from_pickle(vocab_path):

        with open(vocab_path, 'rb') as f:
            vocab = pickle.load(f)
        return vocab

    @staticmethod
    def append_special_tokens(vocab):

        count = 0
        for special_token in Vocabulary.special_tokens:
            if special_token not in vocab.keys():
                count += 1

        Vocabulary.increment_dictionary(vocab, count)

        new_index = 0
        for special_token in Vocabulary.special_tokens:
            if special_token not in vocab.keys():
                my_print(f'Inserting special token {special_token} to vocabulary at position {new_index}.')
                vocab[special_token] = new_index
                new_index += 1

        vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1])}

        return vocab

    @staticmethod
    def increment_dictionary(dictionary, increment):

        assert isinstance(dictionary, dict)
        assert isinstance(increment, int)

        for entry in dictionary:

            assert isinstance(dictionary[entry], int)

            dictionary[entry] += increment
    
    @staticmethod
    def remove_sos_symbol_from_vocabs(vocab):
        if "<S>" in vocab:
            del vocab["<S>"]

        return vocab

    def print_vocab(self):
        for k,v in self.vocab.items():
            my_print(k,v)

    def tokenize(self, inp):
        
        if isinstance(inp, list):
            return self.tokenize_list(inp)
        elif isinstance(inp, str):
            return self.tokenize_word(inp)
        else:
            assert True == False, 'Got unknown input for tokenization.'

    def tokenize_list(self, line):

        assert isinstance(line, list)

        tokenized = []
        for word in line:
            tokenized.append(self.tokenize_word(word))

        return tokenized

    def tokenize_word(self, word):

        if word in self.vocab:
            return int(self.vocab[word])
        else:
            return int(self.vocab[Vocabulary.UNK])


    def detokenize(self, inp):
        
        if isinstance(inp, list):
            return self.detokenize_list(inp)
        elif isinstance(inp, str):
            return self.detokenize_word(inp)
        else:
            assert True == False, 'Got unknown input for detokenization.'

    def detokenize_list(self, line):

        assert isinstance(line, list)

        detokenized = []
        for word in line:
            detokenized.append(self.detokenize_word(word))

        return detokenized

    def detokenize_word(self, word):

        return self.vocab_rev[word]

    
    def remove_padding(self, sentence):
        ret = []
        for word in sentence:
            if word != Vocabulary.PAD:
                ret.append(word)
        return ret


class Dataset:

    def __init__(self):

        self.corpus_size        = 0
        self.worker_corpus_size = 0
        self.data               = [] # [data] Only used when the dataset is loaded into memory
        self.data_ptrs          = [[]] # [[(data_ptr, size)]]

    @staticmethod
    def create_dataset_from_config(config, name, src_path, tgt_path, vocab_src, vocab_tgt, epoch_split=1):

        if config['dataset'] == 'TranslationDataset':
            dataset = TranslationDataset(
                vocab_src   = vocab_src, 
                vocab_tgt   = vocab_tgt,
                src_path    = src_path,
                tgt_path    = tgt_path,
                name        = name,
                maxI        = config['max_sentence_length'],
                in_memory   = config['load_datset_in_memory'],
                epoch_split = epoch_split
            )
        else:
            assert True == False, f'Unrecognized dataset option {config["dataset"]}'

        dataset.load_data_ptrs()

        if dataset.epoch_split > 1:
            dataset.apply_epoch_split()

        assert sum([len(x) for x in dataset.data_ptrs]) == dataset.corpus_size
        assert len(dataset.data_ptrs)                   == dataset.epoch_split

        if Globals.get_number_of_workers() > 1:
            dataset.assign_to_worker(hvd.rank())

        assert sum(hvd.allgather_object(dataset.worker_corpus_size, name='allgather_worker_corpus_size')) == dataset.corpus_size

        if dataset.in_memory:
            dataset.load_data_to_memory()

        return dataset

    def load_data_ptrs(self):
        raise NotImplementedError

    def apply_epoch_split(self):

        RandomState(seed=Globals.get_global_seed()).shuffle(self.data_ptrs)
        
        size        = self.corpus_size // self.epoch_split
        splitted    = []

        for i in range(self.epoch_split-1):

            lower = i * size
            upper = lower + size
            
            splitted.append(self.data_ptrs[0][lower:upper])

            assert len(splitted[-1]) == size

        splitted.append(self.data_ptrs[0][upper:])

        self.data_ptrs = splitted

    def assign_to_worker(self, worker_rank):
        
        self.__sort_by_size()

        workers         = Globals.get_number_of_workers()
        kept_data_ptrs  = []

        for i in range(self.epoch_split):
            
            kept_data_ptrs.append([])

            for j in range(len(self.data_ptrs[i])):

                if j % workers == worker_rank:

                    kept_data_ptrs[i].append(self.data_ptrs[i][j])

        self.data_ptrs = kept_data_ptrs

        self.worker_corpus_size = sum(len(epoch_split) for epoch_split in self.data_ptrs)

        print(f'Worker {worker_rank} is working on {self.worker_corpus_size} sentences for dataset {self.name}', flush=True)

    def __sort_by_size(self):
        for i in range(self.epoch_split):
            self.data_ptrs[i].sort(key=lambda item : item[1])

    def load_data_to_memory(self):
        raise NotImplementedError

    def ptrs_to_tensor(self, ptrs):
        raise NotImplementedError


class TranslationDataset(Dataset):

    def __init__(self, **kwargs):
        super(TranslationDataset, self).__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

    def load_data_ptrs(self):

        discarded_count = 0

        with open(self.src_path, "r") as src_file, open(self.tgt_path, "r") as tgt_file:
            
            for i, (src, tgt) in enumerate(zip(src_file, tgt_file)):
                
                _ , keep, size = self.__preprocess(src, tgt)

                if keep:
                    self.data_ptrs[0].append((i, size))
                else:
                    discarded_count += 1
        
        if discarded_count > 0:
            my_print(f'Discarded {discarded_count} sentences of dataset "{self.name}" because they exceeded the maximum sentence length of {self.maxI}')

        self.corpus_size = len(self.data_ptrs[0])
        self.worker_corpus_size = len(self.data_ptrs[0])

    def __preprocess(self, src, tgt):

        src = src.strip().replace("\n", "").split(" ")
        tgt = tgt.strip().replace("\n", "").split(" ")
        src = self.vocab_src.tokenize(src)
        tgt = self.vocab_tgt.tokenize(tgt)

        size = len(tgt)+1
        
        if len(src)+2 <= self.maxI and size <= self.maxI:

            keep = True

        else:

            if not Globals.is_training():
                
                src = self.vocab_src.tokenize([Vocabulary.UNK for _ in range(5)])
                tgt = self.vocab_tgt.tokenize([Vocabulary.UNK for _ in range(5)])
                keep = True

            else:

                keep = False
        
        return (src, tgt), keep, size

    def load_data_to_memory(self):

        my_print(f'{self.name}: Loading data to memory!')

        indices_list = []
        for ptrs in self.data_ptrs:
            for i in ptrs:
                indices_list.append(i[0])

        indices_list = set(indices_list)
        
        with open(self.src_path, "r") as src_file, open(self.tgt_path, "r") as tgt_file:
            
            for i, (src, tgt) in enumerate(zip(src_file, tgt_file)):
                
                (src, tgt) , keep, _ = self.__preprocess(src, tgt)

                if keep and (i in indices_list):

                    self.data.append((src, tgt))
                
                else:

                    self.data.append(([], []))

        my_print(f'{self.name}: Success: Loaded {len(self.data)} sentences to memory!')
        
    
    def ptrs_to_tensor(self, ptrs):

        pad_index       = self.vocab_src.PAD
        eos_index_src   = self.vocab_src.EOS
        eos_index_tgt   = self.vocab_tgt.EOS

        src, tgt = self.__load_data(ptrs)

        max_s = max([len(x) for x in src])
        max_t = max([len(x) for x in tgt])

        s = [[eos_index_src] + x + [eos_index_src] + [pad_index] * (max_s - len(x)) for x in src]
        t = [[eos_index_tgt] + x + [pad_index] * (max_t - len(x)) for x in tgt]
        o = [x + [eos_index_tgt] + [pad_index] * (max_t - len(x)) for x in tgt]

        s = torch.tensor(s)
        t = torch.tensor(t)
        o = torch.tensor(o)

        return (s, t, o)

    def __load_data(self, ptrs):
        if self.in_memory:
            return self.__load_data_from_memory(ptrs)
        else:
            return self.__load_data_from_file(ptrs)

    def __load_data_from_memory(self, ptrs):
        
        src = []
        tgt = []

        for ptr in ptrs:

            s = self.data[ptr][0]
            t = self.data[ptr][1]

            assert len(s) > 0
            assert len(t) > 0

            src.append(s)
            tgt.append(t)

        return src, tgt

    def __load_data_from_file(self, ptrs):
        raise NotImplementedError('TODO')


class BatchGenerator:

    def __init__(self, **kwargs):

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.threaded   = False
        self.ready      = SharedMemory(False)
        self.done       = SharedMemory(False)
        self.next_batch = SharedMemory([])

    @staticmethod
    def create_batch_generator_from_config(config, dataset, algorithm_class, chunking=None):
        
        algorithm = algorithm_class.create_batch_algorithm_from_config(config, dataset)

        batch_generator = BatchGenerator(
            dataset = dataset,
            algorithm = algorithm,
            chunking = chunking
        )

        return batch_generator


    def start(self):
        self.threaded = True
        self.data_thread = Thread(self.__in_thread_generate_batches, daemon=True, name=f'thread_{self.dataset.name}')
        self.data_thread.start()

    def generate_batches(self):
        """
        A wrapper function around the batch generation that handles threading or no threading.
        In threading mode it will just wait for data that is prepared in the background.
        Otherwise, it will call the batch generation directly.
        """

        if self.threaded:

            while not self.done.read():
                while not self.ready.read() and not self.done.read():
                    time.sleep(0.01)

                if self.ready.read() and not self.done.read():
                    yield self.read_next_batch()

            self.done.write(False)

        else:
            yield from self.algorithm.generate_batches(chunking=self.chunking)

    def __in_thread_generate_batches(self, exit_req):
        """
        Called in seperate thread. Prepares the data and stores them into next_batch.
        """

        for data in self.algorithm.generate_batches(chunking=self.chunking):

            self.next_batch.write(data)
            self.ready.write(True)

            while self.ready.read() and not exit_req.read():
                time.sleep(0.01)

            if exit_req.read():
                break

        self.done.write(True)

    def stop(self):
        self.threaded = False
        self.data_thread.stop()

    def read_next_batch(self):
        next_batch = self.next_batch.read()
        self.ready.write(False)
        return next_batch


class BatchAlgorithm:

    def __init__(self, dataset, chunking=None):
        self.dataset            = dataset
        self.epoch_split_index  = 0

    def generate_batches(self, chunking=None):
        """
        Yields the batch generation result: idx, tensors, total_number_of_steps.
        If chunking is enabled it will return [idx], [tensors], total_number_of_steps
        """

        self.setup()

        batches_ptrs = self.prepare_batches()
        total_steps = len(batches_ptrs)

        if chunking is not None:
            assert isinstance(chunking, int) and chunking >= 1
            chunk_batch_ptrs = []
            chunk_tensors = [[] for _ in self.dataset.ptrs_to_tensor(batches_ptrs[0])]
            total_steps = total_steps // chunking

        for batch_ptrs in batches_ptrs:
            tensors = self.dataset.ptrs_to_tensor(batch_ptrs)
            if chunking is None:
                yield batch_ptrs, tensors, total_steps
            else:

                if len(chunk_batch_ptrs) < chunking:
                    chunk_batch_ptrs.append(batch_ptrs)
                    for i, tensor in enumerate(tensors):
                        chunk_tensors[i].append(tensor)

                    if len(chunk_batch_ptrs) == chunking:
                        yield chunk_batch_ptrs, (tuple(chunk_tensor) for chunk_tensor in chunk_tensors), total_steps
                        chunk_batch_ptrs = []
                        chunk_tensors = [[] for _ in self.dataset.ptrs_to_tensor(batches_ptrs[0])]

                else:
                    raise RuntimeError(f'The actual chunk size {len(chunk_batch_ptrs)} should never exceed the desired one {chunking}')

        self.epoch_split_index = (self.epoch_split_index + 1) % self.dataset.epoch_split


class BucketingBatchAlgorithm(BatchAlgorithm):

    def __init__(self, dataset, **kwargs):
        super(BucketingBatchAlgorithm, self).__init__(dataset)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.random_state = RandomState(seed=Globals.get_global_seed())

        self.bucket_boundaries = [] # type: [int]
        self.buckets = {}           # type: {bucket_idx : [(target_len, sentence_idx)] }

    @staticmethod
    def create_batch_algorithm_from_config(config, dataset):

        batch_algorithm = BucketingBatchAlgorithm(
            dataset,
            batch_size = config['batch_size']
        )

        batch_algorithm.__calculcate_bucket_boundaries()

        return batch_algorithm

    def __calculcate_bucket_boundaries(self):
        self.bucket_boundaries = [float(i*10) for i in range((self.dataset.maxI // 10)+1)]
        self.bucket_boundaries.append(float('inf'))

    def setup(self):
        self.__fill_buckets()

    def __fill_buckets(self):

        self.buckets = {i: [] for i in range(len(self.bucket_boundaries)-1)}

        for (data_ptr, size) in self.dataset.data_ptrs[self.epoch_split_index]:

            lb = 0

            for j in range(len(self.bucket_boundaries)-1):
                lb = self.bucket_boundaries[j]
                ub = self.bucket_boundaries[j+1]
                
                if lb <= size <= ub:
                    self.buckets[j].append((data_ptr, size))
                    break

        sum = 0
        for i in self.buckets:
            sum += len(self.buckets[i])

        assert sum == len(self.dataset.data_ptrs[self.epoch_split_index])

    def prepare_batches(self):
        
        batch_size = self.batch_size
        buckets = self.buckets.copy()

        self.__shuffle(buckets)

        for i in list(buckets):
            if len(buckets[i]) == 0:
                buckets.pop(i)

        batches_ptrs = []

        while len(buckets) != 0:

            i = self.random_state.choice(list(buckets.keys()))

            batch_ptrs = self.__get_batch_indices(i, buckets, batch_size)

            batches_ptrs.append(batch_ptrs)

        return batches_ptrs

    def __shuffle(self, buckets):
        for b in buckets.values():
            self.random_state.shuffle(b)

    def __get_batch_indices(self, bucket_idx, buckets, batch_size):
        
        bucket = buckets[bucket_idx]
        cur_batch_size = 0
        i = 0
        idx = []

        while cur_batch_size < batch_size and i < len(bucket):

            next_batch_size = cur_batch_size + bucket[i][1]

            if next_batch_size <= batch_size or cur_batch_size == 0:
                idx.append(bucket[i][0])
                cur_batch_size = next_batch_size
            else:
                break

            i += 1

        if i == len(bucket):
            buckets.pop(bucket_idx)
        else:
            buckets[bucket_idx] = buckets[bucket_idx][i:]

        return idx


class LinearBatchAlgorithm(BatchAlgorithm):

    def __init__(self, dataset,  **kwargs):
        super(LinearBatchAlgorithm, self).__init__(dataset)

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.data_ptrs = [] # [(target_len, sentence_idx)]

    @staticmethod
    def create_batch_algorithm_from_config(config, dataset):

        batch_algorithm = LinearBatchAlgorithm(
            dataset,
            batch_size = config['batch_size']
        )

        return batch_algorithm

    def setup(self):
        self.__load_data_ptrs()

    def __load_data_ptrs(self):

        self.data_ptrs = []

        for (data_ptr, size) in self.dataset.data_ptrs[self.epoch_split_index]:

            self.data_ptrs.append((data_ptr, size))

        self.data_ptrs.sort(key=lambda item : item[1])

    def prepare_batches(self):

        data_ptrs = self.data_ptrs.copy()

        batches_ptrs = []

        while len(data_ptrs) > 0:

            batch_ptrs, data_ptrs = self.__get_batch_indices(data_ptrs)
            
            batches_ptrs.append(batch_ptrs)

        return batches_ptrs

    def __get_batch_indices(self, data_ptrs):

        batch_size = self.batch_size
        cur_batch_size = 0
        i = 0
        idx = []

        while cur_batch_size < batch_size and i < len(data_ptrs):

            next_batch_size = cur_batch_size + data_ptrs[i][1]

            if next_batch_size <= batch_size or cur_batch_size == 0:
                idx.append(data_ptrs[i][0])
                cur_batch_size = next_batch_size
            else:
                break

            i += 1

        return idx, data_ptrs[i:]
