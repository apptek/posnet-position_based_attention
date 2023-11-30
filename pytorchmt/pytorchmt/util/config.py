import yaml

from pytorchmt.util.debug import my_print

class Config:

    def __init__(self, config):

        self.config = config

    @staticmethod
    def parse_config(args):

        with open(args['config'], 'r') as f:
            
            config = yaml.safe_load(f)
        
        assert config is not None, 'Provided config seems to be empty' 

        for k, v in args.items():

            if v is not None:

                if k in config.keys():

                    my_print(f'CLI config overwrite for "{k}"!')

                config[k] = v

        return Config(config)

    def print_config(self):

        max_key_length = max([len(k) for k in self.config.keys()])

        for key in self.config:

            my_print(key.ljust(max_key_length, '-'), str(self.config[key]).ljust(100, '-'))

    def assert_has_key(self, key):
        assert self.has_key(key), 'Config is missing key "%s"' % (key) 

    def has_key(self, key):
        return key in self.config.keys()

    def __getitem__(self, key):

        default = None

        if isinstance(key, tuple):
            
            assert len(key) == 2

            default = key[1]
            key = key[0]

        if default is None:

            self.assert_has_key(key)
            return self.config[key]

        else:

            if self.has_key(key):
                
                return self.config[key]
            
            else:

                return default

    def __setitem__(self, key, value):

        self.config[key] = value