
def my_print(*args, **kwargs):

    import horovod.torch as hvd

    if hvd.rank() == 0:
        print(*args, flush=True, **kwargs)

def get_number_of_trainable_variables(model):
    sum = 0
    for var in model.trainable_variables:
        cur = 1
        for s in var.shape:
            cur *= s
        sum += cur
    return sum

def print_memory_usage():
    import os
    from resource import getrusage, RUSAGE_SELF
    import torch
    import psutil
    import horovod.torch as hvd
    from pytorchmt.util.globals import Globals

    if Globals.is_gpu():
        current_gpu_memory = torch.cuda.memory_allocated(device=Globals.get_device())
        peak_gpu_memory = torch.cuda.max_memory_allocated(device=Globals.get_device())
    else:
        current_gpu_memory = 0
        peak_gpu_memory = 0

    current_ram = psutil.Process(os.getpid()).memory_info()[0]/2.**30
    peak_ram = getrusage(RUSAGE_SELF).ru_maxrss/10.**6

    print(f'Worker {hvd.rank()}: Memory [Current/Peak]. GPU: [{(current_gpu_memory / 1e9):4.2f}GB/{(peak_gpu_memory / 1e9):4.2f}GB], RAM: [{current_ram:4.2f}GB/{peak_ram:4.2f}GB]', flush=True)

def print_summary(is_train, number, **kwargs):

    import torch
    
    def pr_int(name, value):
        length = len(name) + 2 + 4
        if value is not None:
            return f'{name}: {value:0>4}'
        else:
            return ''.center(length, ' ')

    def pr_float_precise(name, value):
        length = len(name) + 2 + 10 + 6
        if value is not None:
            if value > 1e8:
                value = float('inf')
            return f'{name}: {value:8.6f}'.ljust(length, ' ')
        else:
            return ''.center(length, ' ')

    def pr_float(name, value):
        length = len(name) + 2 + 8 + 2
        if value is not None:
            if value > 1e7:
                value = float('inf')
            return f'{name}: {value:4.2f}'.ljust(length, ' ')
        else:
            return ''.center(length, ' ')

    first_choices = ['train', 'eval']

    if is_train:
        first = first_choices[0]
    else:
        first = first_choices[1]
    first_length = max([len(s) for s in first_choices])

    to_print = (
        f'| {first.center(first_length, " ")} '
        f'| number: {number:0>4} '
    )

    for k, v in kwargs.items():

        if v is None:
            continue

        if isinstance(v, torch.Tensor):
            v = float(v.cpu().detach().numpy())

        if isinstance(v, int):
            to_print = (
                f'{to_print}'
                f'| {pr_int(k, v)} '
            )
        elif isinstance(v, float):
            if v > 1e-2 or v == 0.:
                to_print = (
                    f'{to_print}'
                    f'| {pr_float(k, v)} '
                )
            else:
                to_print = (
                    f'{to_print}'
                    f'| {pr_float_precise(k, v)} '
                )

    my_print(to_print)

def print_allocated_tensors():
    import torch
    import gc
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                my_print(type(obj), obj.size())
        except:
            pass