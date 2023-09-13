import torch


def create_optimizer(model, args):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grounded_parameters = [
        {
            "parames": [
                p for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            args.weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            "weight_decay":
            0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grounded_parameters, lr=args.lr)
    return optimizer