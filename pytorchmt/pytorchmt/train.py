
import argparse
from os.path import join

import torch
import horovod.torch as hvd

from models import create_model_from_config
from pytorchmt.util.config import Config
from pytorchmt.model.scores import Score
from pytorchmt.util.globals import Globals
from pytorchmt.util.trainer import Trainer
from pytorchmt.model.optimizers import Optimizer
from pytorchmt.util.setup import setup_torch_from_config
from pytorchmt.util.debug import my_print, print_memory_usage
from pytorchmt.util.checkpoint_manager import CheckpointManager
from pytorchmt.util.data import Vocabulary, Dataset, BatchGenerator, BucketingBatchAlgorithm, LinearBatchAlgorithm


def parse_cli_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, required=True, 
        help='The path to the config.yaml which contains all user defined parameters.')
    parser.add_argument('--output-folder', type=str, required=True, 
        help='The folder in which to write the training output (ckpts, learning-rates, perplexities etc.)')
    parser.add_argument('--resume-training', type=int, required=False, default=False, 
        help='If you want to resume a training, set this flag to 1 and specify the directory with "resume-training-from".')
    parser.add_argument('--resume-training-from', type=str, required=False, default='', 
        help='If you want to resume a training, specify the output directory here. We expect it to have the same layout as a newly created one.')
    parser.add_argument('--number-of-gpus', type=int, required=False, default=None,
        help='This is usually specified in the config but can also be overwritten from the cli.')

    args = parser.parse_args()

    args.resume_training = bool(args.resume_training)

    return vars(args)

def train(config):

    setup_torch_from_config(config)

    torch.manual_seed(Globals.get_global_seed())

    if config['resume_training']:
        config['output_folder'] = join(config['resume_training_from'])
    
    numbers_dir = join(config['output_folder'], 'numbers')

    vocab_src = Vocabulary.create_vocab(config['vocab_src'])
    vocab_tgt = Vocabulary.create_vocab(config['vocab_tgt'])

    my_print('Vocab Size Src', vocab_src.vocab_size)
    my_print('Vocab Size Tgt', vocab_tgt.vocab_size)
    
    train_dataset = Dataset.create_dataset_from_config(config, 'train_set', config['train_src'], config['train_tgt'], vocab_src, vocab_tgt, epoch_split=config['epoch_split', 1])
    dev_dataset = Dataset.create_dataset_from_config(config, 'dev_set', config['dev_src'], config['dev_tgt'], vocab_src, vocab_tgt)

    train_batch_generator = BatchGenerator.create_batch_generator_from_config(config, train_dataset, BucketingBatchAlgorithm, chunking=config['update_freq'])
    dev_batch_generator = BatchGenerator.create_batch_generator_from_config(config, dev_dataset, LinearBatchAlgorithm)

    if config['threaded_data_loading']:
        train_batch_generator.start()
        dev_batch_generator.start()

    model = create_model_from_config(config, vocab_src, vocab_tgt)
    optimizer = Optimizer.create_optimizer_from_config(config, numbers_dir, model.parameters(), model.named_parameters())

    checkpoint_manager  = CheckpointManager.create_train_checkpoint_manager_from_config(config, model, optimizer)
    checkpoint_manager.restore_or_initialize()

    my_print(f'Trainable variables: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    criterion = Score.create_score_from_config(config)
    trainer = Trainer.create_trainer_from_config(config, train_batch_generator, dev_batch_generator, model, criterion, optimizer, checkpoint_manager, numbers_dir)

    model = model.to(Globals.get_device())
    optimizer = optimizer.to(Globals.get_device())
    criterion = criterion.to(Globals.get_device())

    print_memory_usage()
    my_print(f'Start training at checkpoint {checkpoint_manager.get_checkpoint_number()}!')

    trainer.train()

    if config['threaded_data_loading']:
        train_batch_generator.stop()
        dev_batch_generator.stop()

    if config['average_last_checkpoints', False]:
        checkpoint_manager.average_last_N_checkpoints(config['checkpoints_to_average'])

    if config['average_last_after_best_checkpoints', False]:
        checkpoint_manager.average_N_after_best_checkpoint(config['checkpoints_to_average'])

    average_time_per_checkpoint_s = checkpoint_manager.checkpoint_duration_accum / checkpoint_manager.checkpoint_count
    my_print(f'Average time per checkpoint: {average_time_per_checkpoint_s:4.2f}s {average_time_per_checkpoint_s/60:4.2f}min')

    my_print('Done!')

def start():

    hvd.init()

    my_print(''.center(40, '-'))
    my_print(' Hi! '.center(40, '-'))
    my_print(' Script: train.py '.center(40, '-'))
    my_print(''.center(40, '-'))

    args = parse_cli_arguments()

    config = Config.parse_config(args)
    Globals.set_train_flag(True)
    Globals.set_time_flag(False)
    Globals.set_global_seed(config['seed', 80420])

    config.print_config()

    train(config)


if __name__ == '__main__':

    start()