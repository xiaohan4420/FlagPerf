from transformers import AutoModelForMaskedLM, AutoTokenizer

def create_model(config):
    if config.model_name:
        model = AutoModelForMaskedLM.from_pretrained(config.model_name)
        return model
    else:
        raise ValueError("Invalid model name.")

def create_tokenizer(config):
    if config.model_name:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        return tokenizer
    else:
        raise ValueError("Invalid model name.")