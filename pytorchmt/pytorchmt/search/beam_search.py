
import torch

from pytorchmt.util.debug import my_print
from pytorchmt.util.globals import Globals
from pytorchmt.model.state import StaticState, DynamicState
from pytorchmt.search.search_algorithm import SearchAlgorithm


class BeamSearch(SearchAlgorithm):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @staticmethod
    def create_search_algorithm_from_config(config, model, vocab_src, vocab_tgt):

        return BeamSearch(
            model = model,
            dynamic_states = SearchAlgorithm.get_dynamic_states(model),
            vocab_src = vocab_src,
            vocab_tgt = vocab_tgt,
            D = config['model_dim'],
            maxI = config['max_sentence_length'],
            beam_size = config['beam_size'],
            length_norm = config['length_norm'],
            stepwise = config['stepwise', False]
        )

    def search_batch(self, src):

        B       = src.shape[0]
        srcT    = src.shape[1]
        N       = self.beam_size
        D       = self.D
        BN      = B*N     
        maxI    = self.maxI
        V       = self.V
        PAD     = self.PAD
        EOS     = self.EOS

        fin_storage_scores  = torch.ones(B, N, dtype=torch.float32).to(Globals.get_device()) * -float('inf')
        active_mask         = torch.ones(B, N, dtype=torch.bool).to(Globals.get_device())
        BNa                 = BN
        tgt                 = torch.ones(B, N, 1, dtype=torch.int32).to(Globals.get_device()) * EOS

        precomp_indices = torch.arange(N).to(Globals.get_device()).unsqueeze(0)
        precomp_indices = precomp_indices.repeat(B, 1) + torch.arange(B).to(Globals.get_device()).unsqueeze(1) * N

        tgt_first   = torch.ones(B, 1, dtype=torch.int32).to(Globals.get_device()) * EOS
        
        masks, _    = self.model.create_masks(src, tgt_first, PAD)
        enc_state   = StaticState(self.model.encoder)

        output = self.model.decoder(tgt_first, *enc_state(src, **masks), i=1, **masks)

        output              = output.squeeze(1)
        scores, best_words  = torch.topk(output, k=N, dim=-1)
        best_words          = best_words.unsqueeze(-1)
        tgt                 = torch.cat((tgt, best_words), dim=-1)

        src = src.unsqueeze(1).repeat(1, N, 1)

        enc_state.repeat(N, dim=1)
        self.repeat_batch_dim_of_dynamic_states(N)

        assert list(precomp_indices.shape) == [B, N]
        assert list(scores.shape)          == [B, N]
        assert list(src.shape)             == [B, N, srcT]
        assert list(tgt.shape)             == [B, N, 2]

        i = 2
        while BNa > 0 and i <= maxI:

            srca, tgta   = self.get_active_model_input(src, tgt, active_mask)
            encsa        = enc_state.read(mask_select=active_mask.unsqueeze(-1).unsqueeze(-1))

            self.reduce_states(active_mask)

            assert list(srca.shape) == [BNa, srcT]
            assert list(tgta.shape) == [BNa, i]

            if self.stepwise:
                
                tgta = tgta[:,-1].unsqueeze(-1)
                assert list(tgta.shape) == [BNa, 1]
                
                masks, _    = self.model.create_masks(srca, None, PAD)
                output      = self.model.decoder(tgta, *encsa, i=i, **masks) # [BNa, 1, V]
                output      = output.squeeze(1)

            else:

                masks, _    = self.model.create_masks(srca, tgta, PAD)
                output      = self.model.decoder(tgta, *encsa, **masks) # [BNa, i, V]
                output      = output[:,-1,:]
            
            assert list(output.shape) == [BNa, V]

            output = self.pad_to_N(output, active_mask, precomp_indices)

            assert list(output.shape) == [B, N, V]

            output += scores.unsqueeze(-1)
            output = self.apply_length_norm(output, i)
            output = output.view(B, N*V)

            scores, best_words  = torch.topk(output, k=N, dim=-1)
            best_beams          = torch.div(best_words, V, rounding_mode='floor')
            best_words          = best_words % V

            assert list(scores.shape)     == [B, N]
            assert list(best_beams.shape) == [B, N]
            assert list(best_words.shape) == [B, N]

            tgt, scores, best_words         = self.update_tgt(tgt, active_mask, best_beams, best_words, scores)

            self.reorder_states(best_beams, active_mask)

            assert list(tgt.shape)        == [B, N, i+1]
            assert list(scores.shape)     == [B, N]
            assert list(best_words.shape) == [B, N]

            fin_storage_scores, active_mask = self.update_fin(fin_storage_scores, best_words, active_mask, scores)

            assert list(fin_storage_scores.shape) == [B, N]
            assert list(active_mask.shape)        == [B, N]

            BNa                             = self.update_BNa(active_mask)

            scores = self.remove_length_norm(scores, i)

            i += 1

        tgt = self.select_best_hyp(tgt, fin_storage_scores)

        assert list(tgt.shape) == [B, i]

        if self.stepwise:
            for state in self.dynamic_states:
                state.clear()

        return tgt
    
    def repeat_batch_dim_of_dynamic_states(self, N):
        if self.stepwise:
            for state in self.dynamic_states:
                state.repeat_interleave(N, 0)

    def get_active_model_input(self, src, tgt, active_mask):

        i = tgt.shape[2]
        srcT = src.shape[2]

        active_mask = active_mask.unsqueeze(-1)

        srca = torch.masked_select(src, active_mask).view(-1, srcT)
        tgta = torch.masked_select(tgt, active_mask).view(-1, i)

        return srca, tgta

    def reduce_states(self, active_mask):
        
        if not self.stepwise:
            return
        
        mask = active_mask.view(-1)

        for state in self.dynamic_states:
            state.reduce(mask)

    def pad_to_N(self, output, active_mask, precomp_indices):
        
        B = active_mask.shape[0]
        N = active_mask.shape[1]
        V = output.shape[-1]

        indices = torch.masked_select(precomp_indices, active_mask)
        indices = indices.unsqueeze(-1).repeat(1, V) # [BNa, V]

        res = precomp_indices.to(torch.float32).unsqueeze(-1).repeat(1, 1, V).view(-1, V) # [BN, V]

        output = res.scatter_(dim=0, index=indices, src=output)
        output = output.view(B, N, V)

        output = output.masked_fill_((active_mask == False).unsqueeze(-1), -float('inf'))

        return output

    def apply_length_norm(self, output, i):
        if self.length_norm:
            return output / i
        else:
            return output

    def update_tgt(self, tgt, active_mask, best_beams, best_words, scores):

        i = tgt.shape[-1]
    
        indices         = torch.cumsum(active_mask, dim=-1) - active_mask.to(torch.int32)
        rev_active_mask = active_mask == False

        best_words = torch.gather(best_words, -1, indices)
        best_words = best_words.masked_fill_(rev_active_mask, self.PAD)

        best_beams = torch.gather(best_beams, -1, indices)
        best_beams = torch.where(rev_active_mask, torch.arange(self.beam_size).to(Globals.get_device()), best_beams)

        tgt = torch.gather(tgt, 1, best_beams.unsqueeze(-1).repeat(1, 1, i))
        tgt = torch.cat((tgt, best_words.unsqueeze(-1)), axis=-1)

        scores = torch.gather(scores, -1, indices)
        scores = scores.masked_fill_(rev_active_mask, -float('inf'))

        return tgt, scores, best_words

    def reorder_states(self, best_beams, active_mask):
        """
        best_beams describes how the states are reordered. 
        However, best_beams does not consider finished beams and has N candidates for every batch.
        This also implies that we have a shape mismatch in the first dimension between best_beams and the dynamic
        states, namely [BN] and [BNa]. Therefore, we select from best_beams only entries for the 
        non-finished candiates. Since the dynamic states have the batch and beam dimension flattened, and we
        reorder with respect to indices, we need to make sure that there is no exchange between batches.
        """

        if not self.stepwise:
            return

        N = best_beams.shape[-1]

        # Rescale best_beams within the batch to not count finished beams
        indices = torch.cumsum(active_mask, dim=-1) - active_mask.to(torch.int32)

        best_beams = torch.gather(indices, -1, best_beams.to(torch.int64))

        # Rescale indices to only point to their batch within the flattened BNa dimension
        indices = active_mask.sum(dim=-1)
        indices = torch.cumsum(indices, dim=-1) - indices.to(torch.int32)
        indices = indices.unsqueeze(-1).repeat(1, N)

        best_beams = best_beams + indices

        # Select the best non-finished beams. Within best_beams those are sorted 
        active_mask, _ = torch.sort(active_mask.to(torch.int32), dim= -1, descending=True)
        best_beams = torch.masked_select(best_beams, active_mask.to(torch.bool))

        # Reorder
        for state in self.dynamic_states:
            state.reorder(best_beams)

    def update_fin(self, fin_storage_scores, best_words, active_mask, scores):

        cur_fin_mask       = best_words == self.EOS
        fin_storage_scores = torch.where(cur_fin_mask, scores, fin_storage_scores)

        active_mask = torch.logical_and(cur_fin_mask == False, active_mask)

        return fin_storage_scores, active_mask
    
    def update_BNa(self, active_mask):
        return active_mask.sum()

    def remove_length_norm(self, output, i):
        if self.length_norm:
            return output * i
        else:
            return output

    def select_best_hyp(self, tgt, fin_storage_scores):

        _, indices = torch.topk(fin_storage_scores, k=1)

        tgt = torch.cat([ torch.index_select(a, 0, i) for a, i in zip(tgt, indices) ])

        return tgt