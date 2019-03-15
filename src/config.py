class Config:
    DATA_PATH = "data/toxic_comments"
    PATH = "data/toxic_comments/tmp"
    CLAS_DATA_PATH = PATH+'/class'
    BERT_PRETRAINED_PATH = "bert_pretrained/bert-base-uncased"
    ARGS = {
        "train_size": -1,
        "val_size": -1,
        "full_data_dir": DATA_PATH,
        "data_dir": DATA_PATH,
        "task_name": "toxic_multilabel",
        "no_cuda": False,
        "bert_model": BERT_PRETRAINED_PATH,
        "bert_model_cache": BERT_PRETRAINED_PATH + "/cache",
        "output_dir": CLAS_DATA_PATH + '/output',
        "max_seq_length": 64,
        "do_train": True,
        "do_eval": True,
        "do_lower_case": True,
        "train_batch_size": 8,
        "eval_batch_size": 8,
        "learning_rate": 3e-5,
        "num_train_epochs": 4.0,
        "warmup_proportion": 0.1,
        "no_cuda": False,
        "local_rank": -1,
        "seed": 42,
        "gradient_accumulation_steps": 1,
        "optimize_on_cpu": False,
        "fp16": False,
        "loss_scale": 128
    }