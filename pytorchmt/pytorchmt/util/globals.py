

class Globals:

    __IS_TRAIN  = None
    __TIME      = None
    __WORKERS   = None
    __GPU       = True
    __DEVICE    = None
    __SEED      = None

    @staticmethod
    def set_train_flag(flag):
        if Globals.__IS_TRAIN is None:
            Globals.__IS_TRAIN = flag

    @staticmethod
    def is_training():
        return Globals.__IS_TRAIN

    @staticmethod
    def set_time_flag(flag):
        if Globals.__TIME is None:
            Globals.__TIME = flag

    @staticmethod
    def do_timing():
        return Globals.__TIME

    @staticmethod
    def set_number_of_workers(workers, force=False):
        if Globals.__WORKERS is None or force:
            Globals.__WORKERS = workers

    @staticmethod
    def get_number_of_workers():
        return Globals.__WORKERS

    @staticmethod
    def set_cpu():
        if Globals.__GPU:
            Globals.__GPU = False

    @staticmethod
    def is_gpu():
        return Globals.__GPU

    @staticmethod
    def set_device(device):
        if Globals.__DEVICE is None:
            Globals.__DEVICE = device

    @staticmethod
    def get_device():
        return Globals.__DEVICE

    @staticmethod
    def set_global_seed(seed):
        if Globals.__SEED is None:
            Globals.__SEED = seed
    
    @staticmethod
    def get_global_seed():
        return Globals.__SEED

    @staticmethod
    def increase_global_seed():
        Globals.__SEED += 1

    

