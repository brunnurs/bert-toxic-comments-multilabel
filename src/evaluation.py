import logging
import os

import torch
from sklearn.metrics import roc_curve, auc
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

from src.feature_extraction import convert_examples_to_features
from src.metrics import accuracy_thresh
import numpy as np


class Evaluation:

    def __init__(self, eval_examples, label_list, num_labels, tokenizer, config):
        self.args = config
        self.num_labels = num_labels
        self._initialize(eval_examples, label_list, tokenizer)

    def _initialize(self, eval_examples, label_list, tokenizer):
        os.makedirs(self.args['output_dir'], exist_ok=True)

        eval_features = convert_examples_to_features(eval_examples, label_list, self.args['max_seq_length'], tokenizer)

        logging.info("***** Initializing evaluation *****")
        logging.info("  Num examples = %d", len(eval_examples))
        logging.info("  Batch size = %d", self.args['eval_batch_size'])

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_ids for f in eval_features], dtype=torch.float)

        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)

        self.eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args['eval_batch_size'])

    def evaluate(self, model, device):
        os.makedirs(self.args['output_dir'], exist_ok=True)

        logging.info("***** Running evaluation *****")

        all_logits = None
        all_labels = None

        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for input_ids, input_mask, segment_ids, label_ids in self.eval_dataloader:
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)

            with torch.no_grad():
                tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
                logits = model(input_ids, segment_ids, input_mask)

            #         logits = logits.detach().cpu().numpy()
            #         label_ids = label_ids.to('cpu').numpy()
            #         tmp_eval_accuracy = accuracy(logits, label_ids)
            tmp_eval_accuracy = accuracy_thresh(logits, label_ids)
            if all_logits is None:
                all_logits = logits.detach().cpu().numpy()
            else:
                all_logits = np.concatenate((all_logits, logits.detach().cpu().numpy()), axis=0)

            if all_labels is None:
                all_labels = label_ids.detach().cpu().numpy()
            else:
                all_labels = np.concatenate((all_labels, label_ids.detach().cpu().numpy()), axis=0)

            eval_loss += tmp_eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        #     ROC-AUC calcualation
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        for i in range(self.num_labels):
            fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  #               'loss': tr_loss/nb_tr_steps,
                  'roc_auc': roc_auc}

        output_eval_file = os.path.join(self.args['output_dir'], "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logging.info("  %s = %s", key, str(result[key]))
        #             writer.write("%s = %s\n" % (key, str(result[key])))
        return result
