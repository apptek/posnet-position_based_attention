import threading as th

from pytorchmt.util.debug import my_print

class SharedMemory:

    def __init__(self, value):
        self.value = value
        self.lock = th.Lock()
    
    def read(self):
        self.lock.acquire()
        value = self.value
        self.lock.release()
        return value

    def write(self, value):
        self.lock.acquire()
        self.value = value
        self.lock.release()

class Thread:

    def __init__(self, fn, daemon=True, name='thread'):

        self.fn = fn
        self.exit_req = SharedMemory(False)
        self.name = name

        self.thread = th.Thread(target=self.run, name=name)
        self.thread.daemon = daemon

    def start(self):
        my_print(f'Started {self.name}! Data will be prepared in the background.')
        self.thread.start()

    def run(self):
        while not self.exit_req.read():
            self.fn(self.exit_req)

    def stop(self):
        self.exit_req.write(True)
        my_print(f'Waiting for thread "{self.name}" to terminate!')
        self.thread.join()
        my_print('Terminated!')