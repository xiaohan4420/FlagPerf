import torch
from datasets import load_dataset

class RobertaDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
    
    def __getitem__(self, index):
        example = self.dataset[index]
        text = example["premise"]
        input_ids = self.tokenizer.encode(text, add_special_tokens, padding="max_length")
        return torch.tensor(input_ids)
    
    def __len__(self):
        return len(self.dataset)

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

def prepare_train_dataset(config):
    train_dataset = load_dataset(
        config.dataset_name,
        config.language,
        split=config.train_data,
        cache_dir=config.cache_dir,
    )
    return train_dataset

def process_dataset(dataset, tokenizer):
    return tokenize(
        examples["premise"],
        examples["hypothesis"],
        padding="max_length",
        truncation=True,
    )

def build_train_dataloader(dataset, config):
    train_dataset = dataset
    data_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffel=True,
        batch_size=config.train_batch_size,
    )
    return data_loader