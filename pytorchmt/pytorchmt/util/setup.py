
import torch
import horovod.torch as hvd

from pytorchmt.util.debug import my_print
from pytorchmt.util.globals import Globals


def setup_torch_from_config(config):

    num_gpus_avail = torch.cuda.device_count()

    my_print(f'Number of GPUs available: {num_gpus_avail}')

    my_print('Available devices:', [torch.cuda.get_device_name(i) for i in range(num_gpus_avail)])

    config['number_of_gpus'] = max(0, config['number_of_gpus'])

    assert config['number_of_gpus'] <= num_gpus_avail, f'Not enough GPUs available! Avail: {num_gpus_avail}, Requested {config["number_of_gpus"]}'

    if not Globals.is_training(): # We do not support multi-gpu for search
        config['number_of_gpus'] = min(1, config['number_of_gpus'])

    Globals.set_number_of_workers(max(1, config['number_of_gpus']))

    if config['deterministic', False]:
        torch.use_deterministic_algorithms(True, warn_only=True)

    if config['number_of_gpus'] <= 0:
        my_print('Limiting to CPU!')
        Globals.set_cpu()
        Globals.set_device('cpu')

    else:

        setup_horovod(config)

def setup_horovod(config):

    my_print('Setting up Horovod!')

    config['update_freq'] = config['update_freq'] // Globals.get_number_of_workers()

    my_print(f'Scaled down update_freq to {config["update_freq"]}!')

    torch.cuda.set_device(hvd.local_rank())

    my_print(f'Horovod: Is MPI enabled at runtime? {hvd.mpi_enabled()}! Hvd build with MPI? {hvd.mpi_built()}!')
    my_print(f'Horovod: Hvd compiled with nccl support? {hvd.nccl_built()}! Hvd build with cuda support? {hvd.cuda_built()}!')

    Globals.set_device(f'cuda:{hvd.local_rank()}')