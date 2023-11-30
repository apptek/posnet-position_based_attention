
import time

import torch
import torch.nn as nn

from pytorchmt.util.globals import Globals
from pytorchmt.model.state import StaticState, DynamicState
from pytorchmt.util.debug import my_print, print_memory_usage

class SearchAlgorithm(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

        self.EOS            = self.vocab_tgt.EOS
        self.PAD            = self.vocab_tgt.PAD
        self.V              = self.vocab_tgt.vocab_size

        fin_V_mask = torch.arange(self.V) == self.EOS
        self.fin_V_mask = fin_V_mask.view(1, 1, self.V)


    @staticmethod
    def get_dynamic_states(model):
        
        states = []

        if hasattr(model, "decoder"):

            for name, module in model.decoder.named_modules():
                if isinstance(module, DynamicState):
                    states.append(module)
            
        return states

    def search(self, search_batch_generator, output_file):

        results = []
        step_times = []
        read_order = []
        num_sentences = 0
        tokens_accum = 0

        start = time.perf_counter()

        for idx, (src, _, out), _ in search_batch_generator.generate_batches():
            
            src = src.to(Globals.get_device())
            out = out.to(Globals.get_device())

            num_sentences += src.shape[0]

            start_step = time.perf_counter()

            result = self.search_batch(src)

            end_step = time.perf_counter()
            step_times.append((end_step - start_step))

            src, tokens = self.to_string_list(src, self.vocab_src)
            result, _ = self.to_string_list(result, self.vocab_tgt)
            out, _ = self.to_string_list(out, self.vocab_tgt)

            self.print_search_result(src, result, out)

            results += result
            read_order += idx
            tokens_accum += tokens

        end = time.perf_counter()

        assert num_sentences == search_batch_generator.dataset.corpus_size

        print_memory_usage()

        time_in_s = end - start
        
        my_print(f"Searching batch took: {time_in_s:4.2f}s, {(time_in_s) / 60:4.2f}min")
        my_print(f"Average step time: {sum(step_times)/len(step_times):4.2f}s")
        my_print(f"Decoded {tokens_accum} tokens with {tokens_accum/time_in_s:4.2f} tokens/s")


        self.write_sentences_to_file_in_order(results, read_order, output_file)

        return results

    def search_batch(self, src):
        raise NotImplementedError()

    def to_string_list(self, inps, vocab):

        processed = []
        tokens = 0

        for hyp in inps:

            if isinstance(hyp, torch.Tensor):
                hyp = hyp.cpu().detach().numpy().tolist()

            hyp = vocab.detokenize(hyp)
            hyp = vocab.remove_padding(hyp)

            processed.append(hyp)

            tokens += len(hyp)

        return processed, tokens

    def print_search_result(self, src, result, out):

        for s, r, o in zip(src, result, out):

            my_print('===')
            my_print('src : ', ' '.join(s[1:-1]))
            my_print('hyp : ', ' '.join(r[1:-1]))
            my_print('ref : ', ' '.join(o[:-1]))

    def write_sentences_to_file_in_order(self, results, read_order, output_file):

        with open(output_file, "w") as file:

            for i in range(len(results)):
                
                idx = read_order.index(i)
                tgt = results[idx][1:-1]
                tgt = " ".join(tgt)

                file.write(tgt + "\n")