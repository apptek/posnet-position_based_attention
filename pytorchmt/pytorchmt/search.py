
import argparse
from os.path import join

import torch
import horovod.torch as hvd

from models import create_model_from_config
from pytorchmt.util.config import Config
from pytorchmt.util.globals import Globals
from pytorchmt.util.setup import setup_torch_from_config
from pytorchmt.util.checkpoint_manager import CheckpointManager
from pytorchmt.util.debug import my_print, get_number_of_trainable_variables
from pytorchmt.search.search_algorithm_selector import SearchAlgorithmSelector
from pytorchmt.util.data import Vocabulary, Dataset, BatchGenerator, LinearBatchAlgorithm


def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, 
        help='The path to the config.yaml which contains all user defined parameters. It may or may not match the one trained with. This is up to the user to ensure.')
    parser.add_argument('--checkpoint-path', type=str, required=True, 
        help='The checkpoint.pt file containing the model weights.')
    parser.add_argument('--output-folder', type=str, required=False, default=None, 
        help='The output folder in which to write the score and hypotheses.')
    parser.add_argument('--number-of-gpus', type=int, required=False, default=None, 
        help='This is usually specified in the config but can also be overwritten from the cli. However, in search this can only be 0 or 1. We do not support multi-gpu decoding. If you set it to >1 we will set it back to 1 so that you dont need to modify the config in search.')

    args = parser.parse_args()

    return vars(args)

def search(config):

    setup_torch_from_config(config)

    config['batch_size'] = config['batch_size_search']

    vocab_src = Vocabulary.create_vocab(config['vocab_src'])
    vocab_tgt = Vocabulary.create_vocab(config['vocab_tgt'])

    my_print('Vocab Size Src', vocab_src.vocab_size)
    my_print('Vocab Size Tgt', vocab_tgt.vocab_size)

    search_src = config['dev_src']
    search_tgt = config['dev_tgt']

    if config['search_test_set', False]:
        search_src = config['test_src']
        search_tgt = config['test_tgt']

    search_dataset = Dataset.create_dataset_from_config(config, 'search_set', search_src, search_tgt, vocab_src, vocab_tgt)
    
    search_batch_generator = BatchGenerator.create_batch_generator_from_config(config, search_dataset, LinearBatchAlgorithm)

    if config['threaded_data_loading']:
        search_batch_generator.start()

    model = create_model_from_config(config, vocab_src, vocab_tgt)

    checkpoint_manager = CheckpointManager.create_eval_checkpoint_manager_from_config(config, model)
    checkpoint_manager.restore(config['checkpoint_path'])

    model = model.to(Globals.get_device())

    my_print(f'Trainable variables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    search_algorithm = SearchAlgorithmSelector.create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt)

    ckpt_suffix = config['checkpoint_path'].split('ckpt-')[-1].replace('.pt', '')

    if ckpt_suffix.isdigit():
        output_file = join(config['output_folder'], f'hyps_{checkpoint_manager.checkpoint_count}')
    else:
        output_file = join(config['output_folder'], f'hyps_{ckpt_suffix}')

    my_print(f'Searching checkpoint {checkpoint_manager.checkpoint_count}!')

    model.eval()

    with torch.no_grad():
        
        search_algorithm.search(search_batch_generator, output_file)

    if config['threaded_data_loading']:
        search_batch_generator.stop()

    my_print('Done!')

def start():

    hvd.init()

    my_print(''.center(40, '-'))
    my_print(' Hi! '.center(40, '-'))
    my_print(' Script: search.py '.center(40, '-'))
    my_print(''.center(40, '-'))

    args = parse_cli_arguments()

    config = Config.parse_config(args)
    Globals.set_train_flag(False)
    Globals.set_time_flag(False)

    config.print_config()

    search(config)


if __name__ == '__main__':

    start()