from transformers import AutoModelForMaskedLM, AutoTokenizer

def create_model(config):
    if config.model_name:
        model = AutoModelForMaskedLM.from_pretrained(config.model_name, cache_dir=config.cache_dir)
        return model
    else:
        raise ValueError("Invalid model name.")

def create_tokenizer(config):
    tokenizer_kwargs = {
        "cache_dir": config.cache_dir,
        # other tokenizer kwargs below:
    }

    if config.model_name:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name, **tokenizer_kwargs)
        return tokenizer
    else:
        raise ValueError("Invalid model name.")