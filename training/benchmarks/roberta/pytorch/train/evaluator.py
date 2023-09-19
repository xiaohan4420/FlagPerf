import evaluate
from transformers import (
    DataCollatorForLanguageModeling,
)


def my_collator(config, tokenizer):
    pad_to_multiple_of_8 = config.line_by_line and \
        config.fp16 and not config.pad_to_max_length
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm_probability=config.mlm_probability,
        pad_to_multiple_of=8 if pad_to_multiple_of_8 else None,
    )
    return data_collator

class Evaluator:
    def __init__(self, config, metric_name):
        self.config = config
        self.metric_name = metric_name
        self.metric = evaluate.load(self.metric_name)

    def reset(self):
        self.metric = evaluate.load(self.metric_name)
    
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits.argmax(dim=-1)
    
    def compute_metric(self, eval_preds):
        preds, labels = eval_preds
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return self.metric.compute(
            predictions=preds, 
            references=labels
            )
    