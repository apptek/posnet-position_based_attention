
from pytorchmt.search.beam_search import BeamSearch

class SearchAlgorithmSelector:

    def __init__(self):
        pass

    @staticmethod
    def create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt):
        
        if config['search_algorithm'] == 'Beam':
            
            return BeamSearch.create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt)
        
        else:

            raise ValueError(f'Unrecognized search algorithm {config["search_algorithm"]}')