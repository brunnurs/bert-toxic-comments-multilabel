import argparse
import logging
import os

import torch
from pytorch_pretrained_bert import BertTokenizer

from src.config import Config
from src.data_representation import MultiLabelTextProcessor
from src.evaluation import Evaluation
from src.load_data import load_training_data
from src.model import get_model, BertForMultiLabelSequenceClassification
from src.optimizer import build_optimizer_scheduler
from src.prediction import predict
from src.training import Training

if __name__ == "__main__":

    processor = MultiLabelTextProcessor(Config.DATA_PATH)
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(Config.ARGS['bert_model'],
                                              do_lower_case=Config.ARGS['do_lower_case'])

    training = Training(Config.ARGS)
    device, n_gpu = training.prepare()
    logging.info("initialized device: {}".format(device))

    # Load a trained model that you have fine-tuned
    saved_model_path = os.path.join(Config.ARGS['bert_model_cache'], "finetuned_pytorch_model.bin")
    model_state_dict = torch.load(saved_model_path)
    model = BertForMultiLabelSequenceClassification.from_pretrained(Config.ARGS['bert_model'],
                                                                    num_labels=num_labels,
                                                                    state_dict=model_state_dict)
    model.to(device)

    result = predict(model, Config.DATA_PATH, device, label_list, tokenizer, Config.ARGS)
    print("Result-Shape: {}".format(result.shape))

    cols = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    result[cols].to_csv(os.path.join(Config.DATA_PATH, 'toxic_kaggle_submission_14_single.csv'), index=None)
