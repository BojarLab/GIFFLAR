# import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import copy
from pathlib import Path

from datasets import load_dataset
from jsonargparse import ArgumentParser
from tokenizers import Encoding
import torch
from transformers import TrainingArguments, DataCollatorForLanguageModeling, Trainer, EsmForMaskedLM, EsmConfig

from gifflar.tokenize.pretokenize import GlycoworkPreTokenizer, GrammarPreTokenizer, CharacterPreTokenizer
from gifflar.tokenize.tokenizer import GIFFLARTokenizer
from gifflar.utils import read_yaml_config

# ESM2 model specifications
#
# t6:   6 layers,  320 hidden size, 20 attention heads,   8M parameters
# t12: 12 layers,  480 hidden size, 20 attention heads,  35M parameters
# t30: 30 layers,  640 hidden size, 20 attention heads, 150M parameters
# t33: 33 layers, 1280 hidden size, 20 attention heads, 650M parameters
# t36: 36 layers, 2560 hidden size, 20 attention heads,   3B parameters
# t48: 48 layers, 5120 hidden size, 20 attention heads,  15B parameters


PRE_TOKENIZERS = {
    "lib": GlycoworkPreTokenizer,
    "glyles": GrammarPreTokenizer,
    "char": CharacterPreTokenizer,
}

def train(**kwargs):
    pretokenizer = GrammarPreTokenizer() if kwargs["tokenization"]["pretokenizer"] == "glyles" else GlycoworkPreTokenizer()
    tokenizer = GIFFLARTokenizer(pretokenizer, kwargs["tokenization"]["tokenizer"].upper()).load(kwargs["tokenization"].get("token_file", None))
        
    def tokenize_function(entries: dict) -> Encoding:
        return tokenizer(entries["text"], padding="max_length", truncation=True, max_length=kwargs["max_length"])

    def filtering_function(entries: dict) -> bool:  # filter all invalid entries as they are only one unknonw token
        return len(entries["input_ids"]) > 1 or entries["input_ids"][0] != tokenizer.unk_token_id

    dataset = load_dataset("text", data_files={"train": kwargs["corpus_file"]})
    proc_ds = dataset.map(tokenize_function, batched=False, remove_columns=["text"])
    proc_ds = proc_ds.filter(filtering_function, batched=False)

    esm_config = EsmConfig(
        vocab_size=tokenizer.vocab_size + 7,  # accounting for special tokens
        num_hidden_layers=kwargs["model"]["num_layers"],
        hidden_size=kwargs["model"]["hidden_size"],
        num_attention_heads=kwargs["model"]["num_heads"],
        pad_token_id=tokenizer.pad_token_id,
        max_position_embeddings=tokenizer.vocab_size + kwargs["max_length"],
        position_embedding_type="absolute",  # try: "rotary"
    )

    model = EsmForMaskedLM(
        config=esm_config,
    )# .to("cpu")

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    kwargs["output_dir"] = Path(kwargs["root_dir"]) / kwargs["tokenization"]["name"].lower()
    Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=kwargs["output_dir"],
        num_train_epochs=kwargs["model"]["epochs"],
        per_device_train_batch_size=kwargs["model"]["batch_size"],
        save_strategy="epoch",
        # fp16=True,
        # gradient_checkpointing=True,
        dataloader_num_workers=0,
        # use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
        train_dataset=proc_ds["train"],
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    trainer.train()


def main(config: str | Path):
    custom_args = read_yaml_config(config)
    if isinstance(custom_args["tokenizations"], dict):
        tokenizations = [custom_args["tokenizations"]]
    else:
        tokenizations = custom_args["tokenizations"]
    del custom_args["tokenizations"]

    for token in tokenizations:
        tmp_config = copy.deepcopy(custom_args)
        tmp_config["tokenization"] = token
        # try:
        train(**tmp_config)
        # except Exception as e:
        #     print(f"Error: {e} occurred during training with config: {tmp_config}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config file")
    main(parser.parse_args().config)
