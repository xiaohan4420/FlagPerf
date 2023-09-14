import torch
from datasets import load_dataset


# class RobertaDataset(torch.utils.data.Dataset):
#     def __init__(self, dataset, tokenizer):
#         self.dataset = dataset
#         self.tokenizer = tokenizer
    
#     def __getitem__(self, index):
#         example = self.dataset[index]
#         text = example["premise"]
#         input_ids = self.tokenizer.encode(text, add_special_tokens, padding="max_length")
#         return torch.tensor(input_ids)
    
#     def __len__(self):
#         return len(self.dataset)

def prepare_raw_dataset(config):
    raw_dataset = load_dataset(
        config.dataset_name,
        config.dataset_config_name,
        cache_dir=config.cache_dir,
    )
    return raw_dataset
    # if "validation" not in raw_dataset.keys():
    #     raw_dataset["validation"] = load_dataset(
    #         config.dataset_name,
    #         config.dataset_config_name,
    #         split=f"train[:{config.}]"
    #     )

def preprocessing_datasets(config, tokenizer, raw_datasets):
    column_names = list(raw_datasets["train"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]
    padding = "max_length" if config.pad_to_max_length else False

    def tokenize_function(examples):
        # Remove empty lines
        examples[text_column_name] = [
            line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
        ]
        return tokenizer(
            examples[text_column_name],
            padding=padding,
            truncation=True,
            # max_length=max_seq_length,
            return_special_tokens_mask=True,  
        )
    
    tokenized_dataset = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=config.preprocessing_num_workers,
        remove_columns=[text_column_name],
        # load_from_cache_file=not config.overwrite_cache,
        desc="Running tokenizer on dataset line by line.",
    )

    return tokenized_dataset

def prepare_train_dataset(config, tokenized_datasets):
    if config.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if config.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), config.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
    return train_dataset

def prepare_eval_dataset(config, tokenized_datasets):
    if config.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = tokenized_datasets["validation"]
        if config.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), config.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
    return eval_dataset