import torch
from torch.utils.data import TensorDataset, RandomSampler, DistributedSampler, DataLoader

from src.feature_extraction import convert_examples_to_features
import logging


def load_training_data(train_examples, label_list, tokenizer, args):
    train_features = convert_examples_to_features(train_examples, label_list, args['max_seq_length'], tokenizer)

    logging.info("***** Load Training Data *****")
    logging.info("  Num examples = %d", len(train_examples))
    logging.info("  Batch size = %d", args['train_batch_size'])
    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in train_features], dtype=torch.float)
    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    if args['local_rank'] == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)

    return DataLoader(train_data, sampler=train_sampler, batch_size=args['train_batch_size'])
