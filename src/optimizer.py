from pytorch_pretrained_bert import BertAdam

from src.cyclic_learning_rate_scheduler import CyclicLR


def build_optimizer_scheduler(model, num_train_steps, args):
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args['fp16']:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args['learning_rate'],
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args['loss_scale'] == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args['loss_scale'])

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args['learning_rate'],
                             warmup=args['warmup_proportion'],
                             t_total=num_train_steps)

    scheduler = CyclicLR(optimizer, base_lr=2e-5, max_lr=5e-5, step_size=2500, last_batch_iteration=0)

    return optimizer, scheduler
