import math

import torch
import horovod.torch as hvd

from pytorchmt.model.scores import Score
from pytorchmt.util.globals import Globals
from pytorchmt.util.timer import Timer, ContextTimer
from pytorchmt.util.debug import my_print, print_summary, print_memory_usage


class Trainer:

    def __init__(self, **kwargs):
        super(Trainer, self).__init__()

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def create_trainer_from_config(config, train_batch_generator, dev_batch_generator, model, criterion, optimizer, checkpoint_manager, numbers_dir):
        
        pad_index = train_batch_generator.dataset.vocab_src.PAD

        assert pad_index == train_batch_generator.dataset.vocab_tgt.PAD == dev_batch_generator.dataset.vocab_tgt.PAD == dev_batch_generator.dataset.vocab_src.PAD

        trainer = Trainer(
            train_batch_generator = train_batch_generator,
            dev_batch_generator = dev_batch_generator,
            model = model,
            criterion = criterion,
            optimizer = optimizer,
            checkpoint_manager = checkpoint_manager,
            numbers_dir = numbers_dir,
            pad_index = pad_index,
            batch_size = config['batch_size'],
            update_freq = config['update_freq'],
            max_sentence_length = config['max_sentence_length'],
            allow_none_type_gradients = config['allow_none_type_gradients', False],
            deterministic = config['deterministic', False]
        )

        return trainer

    def train(self):

        self.model.train()
        self.model.zero_grad()

        self.checkpoint_manager.timer_start()

        while self.checkpoint_manager.keep_going():
            
            step = 1
            max_steps = None

            for _, (src, tgt, out), total_steps in self.train_batch_generator.generate_batches():

                assert len(src) == len(tgt) == len(out) == self.update_freq

                if max_steps is None:
                    max_steps = min(hvd.allgather_object(total_steps, name=f'gather_steps_train'))
                elif step > max_steps:
                    break

                _, L = self.train_step(src, tgt, out)

                self.optimizer.write_lr_to_file(L)

                if self.checkpoint_manager.do_checkpoint_after_step():
                    self.do_checkpoint()

                    if not self.checkpoint_manager.keep_going():
                        return

                step += 1

            if self.checkpoint_manager.do_checkpoint_after_epoch():
                self.do_checkpoint()
    
    def do_checkpoint(self):

        time_passed_s = self.checkpoint_manager.timer_end()
        checkpoint_number = self.checkpoint_manager.get_checkpoint_number()

        train_ce, train_ce_smooth   = self.criterion.average_and_reset()
        train_ppl, train_ppl_smooth = self.__calculate_ppl(train_ce, train_ce_smooth)

        to_print = {
            'ce':               train_ce,
            'ce_smooth':        train_ce_smooth,
            'ppl':              train_ppl,
            'ppl_smooth':       train_ppl_smooth,
            'train_steps':      self.checkpoint_manager.step_count-1
        }

        print_summary(True, checkpoint_number, **to_print)

        dev_ppl = self.eval(checkpoint_number)

        print_memory_usage()
        my_print(f'Training checkpoint took: {time_passed_s:4.2f}s, {time_passed_s / 60:4.2f}min')

        self.checkpoint_manager.save(dev_ppl)

        Score.write_score_to_file(self.numbers_dir, 'train_ppl', train_ppl)
        Score.write_score_to_file(self.numbers_dir, 'dev_ppl',   dev_ppl)

        if Globals.do_timing():
            model_time = Timer.print_timing_summary(self.model)
            ContextTimer.print_summary(model_time)

        self.checkpoint_manager.timer_start()

    def eval(self, checkpoint_number):

        self.model.eval()
        self.model.zero_grad()

        step        = 0
        max_steps   = None

        for _, (src, tgt, out), total_steps in self.dev_batch_generator.generate_batches():

            if max_steps is None:
                max_steps = min(hvd.allgather_object(total_steps, name=f'gather_steps_eval'))
            elif step > max_steps:
                break

            ce, ce_smooth, _ = self.eval_step(src, tgt, out)

            step += 1

        ce, ce_smooth       = self.criterion.average_and_reset()
        ppl, ppl_smooth     = self.__calculate_ppl(ce, ce_smooth)

        to_print = {
            'ce':               ce,
            'ce_smooth':        ce_smooth,
            'ppl':              ppl,
            'ppl_smooth':       ppl_smooth,
            'eval_steps':       step
        }

        print_summary(False, checkpoint_number, **to_print)

        self.model.train()
        self.model.zero_grad()

        return ppl

    def __calculate_ppl(self, ce, ce_smooth):

        try: 
            ppl = math.exp(ce) 
            ppl_smooth = math.exp(ce_smooth)

        except OverflowError: 
            ppl = float('inf')
            ppl_smooth = float('inf')

        return ppl, ppl_smooth

    def train_step(self, src, tgt, out):

        L_accum = 0

        for i in range(len(src)):
            L = self.train_ministep(src[i], tgt[i], out[i])
            L_accum += L

        with ContextTimer('average_gradients'):
            for p in self.model.parameters():
                if p.grad is not None:
                    p.grad /= L_accum
                else:
                    if not self.allow_none_type_gradients:
                        raise RuntimeError(f'Detected NoneType gradient!')

        with ContextTimer('optimizer_step'):
            lr = self.optimizer.step()
        
        self.model.zero_grad()

        return (lr, L_accum)
    
    def train_ministep(self, src, tgt, out):

        src = src.to(Globals.get_device())
        tgt = tgt.to(Globals.get_device())
        out = out.to(Globals.get_device())

        with ContextTimer('model_mask_creation'):
            masks, out_mask = self.model.create_masks(src, out, self.pad_index)

        output, _ = self.model(src, tgt, **masks)

        with ContextTimer('criterion'):
            _, ce_smooth, L_ce = self.criterion(output, out, out_mask=out_mask)

        with ContextTimer('backpropagation'):
            ce_smooth.backward()

        if self.deterministic:
            Globals.increase_global_seed()
            torch.manual_seed(Globals.get_global_seed())

        return L_ce

    def eval_step(self, src, tgt, out):
        
        src = src.to(Globals.get_device())
        tgt = tgt.to(Globals.get_device())
        out = out.to(Globals.get_device())

        with torch.no_grad():
            masks, out_mask     = self.model.create_masks(src, out, self.pad_index)
            output, _           = self.model(src, tgt, **masks)
            ce, ce_smooth, L_ce = self.criterion(output, out, out_mask=out_mask)

        return ce, ce_smooth, L_ce