import argparse
import logging

import torch
from pytorch_pretrained_bert import BertTokenizer

from src import Training
from src.config import Config
from src.data_representation import MultiLabelTextProcessor
from src.evaluation import Evaluation
from src.load_data import load_training_data
from src.model import get_model
from src.optimizer import build_optimizer_scheduler

if __name__ == "__main__":
    processor = MultiLabelTextProcessor(Config.DATA_PATH)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(Config.ARGS['bert_model'],
                                              do_lower_case=Config.ARGS['do_lower_case'])

    train_examples = processor.get_train_examples(Config.ARGS['full_data_dir'], size=Config.ARGS['train_size'])
    logging.info("loaded training examples: {}".format(len(train_examples)))

    num_train_steps = int(len(train_examples) / Config.ARGS['train_batch_size'] / Config.ARGS['gradient_accumulation_steps'] * Config.ARGS['num_train_epochs'])

    model = get_model(Config.ARGS, num_labels)
    logging.info("initialized pretrained model")

    training = Training(Config.ARGS)
    device, n_gpu = training.prepare()
    logging.info("initialized device: {}".format(device))

    if Config.ARGS['fp16']:
        model.half()

    model.to(device)

    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    optimizer, scheduler = build_optimizer_scheduler(model, num_train_steps, Config.ARGS)
    logging.info("Built optimizer: {} and scheduler: {}".format(optimizer, scheduler))

    eval_examples = processor.get_dev_examples(Config.ARGS['data_dir'], size=Config.ARGS['val_size'])
    evaluation = Evaluation(eval_examples, label_list, num_labels, tokenizer, Config.ARGS)
    logging.info("loaded and initialized evaluation examples {}".format(len(eval_examples)))

    train_data_loader = load_training_data(train_examples, label_list, tokenizer, Config.ARGS)
    logging.info("loaded training data.")

    model.unfreeze_bert_encoder()
    logging.info("Bert encoder unfreezed")

    training.fit(Config.ARGS['num_train_epochs'], num_train_steps, train_data_loader, model, optimizer, evaluation)

    model.save(Config.ARGS['bert_model_cache'])

