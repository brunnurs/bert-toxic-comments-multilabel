import logging
import random

import numpy as np
import torch

from tqdm import tqdm


class Training:

    def __init__(self, config):
        self.args = config

    def fit(self, num_epocs, num_train_steps, train_dataloader, model, optimizer, evaluation):
        global_step = 0
        model.train()
        logging.info("Start training!")
        for i_ in tqdm(range(int(num_epocs)), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):

                batch = tuple(t.to(self.device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, label_ids)
                if self.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if self.args['gradient_accumulation_steps'] > 1:
                    loss = loss / self.args['gradient_accumulation_steps']

                if self.args['fp16']:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % self.args['gradient_accumulation_steps'] == 0:
                    #             scheduler.batch_step()
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = self.args['learning_rate'] * self.warmup_linear(global_step / num_train_steps,
                                                                                   self.args['warmup_proportion'])
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            logging.info('Loss after epoc {}'.format(tr_loss / nb_tr_steps))
            logging.info('Eval after epoc {}'.format(i_ + 1))
            evaluation.evaluate(model, self.device)

    def prepare(self):
        self.device, self.n_gpu = self._setup_gpu_with_torch()

        self.args['train_batch_size'] = int(self.args['train_batch_size'] / self.args['gradient_accumulation_steps'])

        self._init_seed_everywhere(self.n_gpu)

        return self.device, self.n_gpu

    def _init_seed_everywhere(self, n_gpu):
        random.seed(self.args['seed'])
        np.random.seed(self.args['seed'])
        torch.manual_seed(self.args['seed'])
        if n_gpu > 0:
            torch.cuda.manual_seed_all(self.args['seed'])

    def _setup_gpu_with_torch(self):
        # Setup GPU parameters
        if self.args["local_rank"] == -1 or self.args["no_cuda"]:
            device = torch.device("cuda" if torch.cuda.is_available() and not self.args["no_cuda"] else "cpu")
            n_gpu = torch.cuda.device_count()
        #     n_gpu = 1
        else:
            torch.cuda.set_device(self.args['local_rank'])
            device = torch.device("cuda", self.args['local_rank'])
            n_gpu = 1
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend='nccl')
        logging.info("device we use: '{}' numbers of gpu: '{}', distributed training: '{}', 16-bits training: '{}'"
                     .format(device, n_gpu, bool(self.args['local_rank'] != -1), self.args['fp16']))
        return device, n_gpu

    @staticmethod
    def warmup_linear(x, warmup=0.002):
        if x < warmup:
            return x / warmup
        return 1.0 - x
