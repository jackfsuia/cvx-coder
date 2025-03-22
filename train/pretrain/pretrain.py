# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.
# This code is based on the revised code from fastchat based on tatsu-lab/stanford_alpaca.


from dataclasses import dataclass, field
import json
import time
import logging
import os
from typing import Dict, Optional, List,Union
import torch
from torch.utils.data import Dataset
import re
import torch
from torch import nn
from torch.utils.data import DataLoader
import transformers
from transformers import Trainer, GPTQConfig, deepspeed,AutoModel
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training,PeftModel
from accelerate.utils import DistributedType
from datasets import load_dataset
import safetensors.torch
from transformers import PreTrainedModel,Qwen2ForCausalLM
import math
from tqdm import tqdm
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="/hy-tmp/Qwen-7B-Chat")
    model_checkpoint_bin: Optional[str] = None

@dataclass
class DataArguments:
    data_path: str = field(
        default="/hy-tmp/train_data.jsonl", metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=500,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = False
    output_dir: str ="/hy-tmp/"
    resume_from_checkpoint: Optional[str] = None
    evaluate_before_train:bool = False
@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj","up_proj","down_proj"]
    )
    modules_to_save:List[str] = field(
        default_factory=lambda:["embed_tokens", "lm_head"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
@dataclass
class DebugArguments:
    debugpy: bool = False

parser = transformers.HfArgumentParser(
    (ModelArguments, DataArguments, TrainingArguments, LoraArguments, DebugArguments)
)
(
    model_args,
    data_args,
    training_args,
    lora_args,
    debugpy_args,
) = parser.parse_args_into_dataclasses()

if debugpy_args.debugpy:
    import debugpy
    try:
        debugpy.listen(("localhost", 9501))
        ("Waiting for debugger attach")
        debugpy.wait_for_client()
    except Exception as e:
        pass


IGNORE_TOKEN_ID =-100 # LabelSmoother.ignore_index


local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def tokenize_function_all(item, tokenizer, max_len):

    full = tokenizer.apply_chat_template(item['messages'], tokenize=False, add_generation_prompt=False)
    # full=f'<|user|>\n{item["input"]}<|end|>\n<|assistant|>\n{item["response"]}<|end|><|endoftext|>'

    it=tokenizer(full,truncation=False,padding=False)
#    tokenizer.eos_token_id
    it["input_ids"]=it["input_ids"]+[tokenizer.pad_token_id]
    it=it["input_ids"][:max_len]
    useful_len=len(it)
    useful_seq=it
    it=it+[tokenizer.pad_token_id]*(max_len-useful_len)
    # if useful_len>1400:
    #     print(useful_len)
    full_text = {}
    full_text["input_ids"] = it
    full_text["attention_mask"] = useful_len*[1]+[0]*(max_len-useful_len)
    full_text["label"] =  useful_seq+[IGNORE_TOKEN_ID]*(max_len-useful_len)
    full_text["labels"] = useful_seq+[IGNORE_TOKEN_ID]*(max_len-useful_len)

    return full_text


def tokenize_function_part(item, tokenizer, max_len):
#bufen
    inputs=f'<|user|>\n{item["input"]}<|end|>\n<|assistant|>\n'

    t_inputs=tokenizer(inputs,truncation=False)

    len_inputs=len(t_inputs["input_ids"])

    outputs=f'{item["response"]}<|end|><|endoftext|>'
    t_outputs=tokenizer(outputs,truncation=False)

    
    it=t_inputs["input_ids"] + t_outputs["input_ids"][1:]
    it=it[:max_len]
    useful_len=len(it)

    print(useful_len,'---')
    print(len_inputs,'+',len(t_outputs["input_ids"][1:]))

    it=it+[tokenizer.eos_token_id]*(max_len-useful_len)


    full_text = {}
    full_text["input_ids"] = it
    full_text["attention_mask"] = useful_len*[1]+[0]*(max_len-useful_len)
    # target all sequence

    full_text["label"] =  [IGNORE_TOKEN_ID]*len_inputs+t_outputs["input_ids"][1:]+[IGNORE_TOKEN_ID]*(max_len-useful_len)
    full_text["label"] = full_text["label"][:max_len]
    full_text["labels"] = full_text["label"]
    print(len(full_text["input_ids"]))
    print(len(full_text["attention_mask"]))
    print(len(full_text["label"]))


    return full_text

def print_trainable_parameters(model):
    r"""
    Returns the number of trainable parameters and the number of all parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    print("----traniable---below-")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'trainable:{name}')
        else:
            print(f'untrainable:{name}')
    print("----traniable--above--")


class pret_dataset(Dataset):
    def __init__(self, path, tokenizer, seq_len) -> None:
        super().__init__()
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
            token_list = tokenizer(txt, truncation=False)["input_ids"][1:]
            print(token_list[0:3])
            self.start_token = 1
        step = int(seq_len / 2)

        self.data = []

        tk_size = len(token_list)

        back_seq_len = seq_len - 1


        prefix = [1]#"According to the CVX Users’ Guide, "
        for seq_id in tqdm(range(math.ceil(tk_size / step))):

            seq = token_list[seq_id * step : seq_id * step + back_seq_len]
            
            seq = prefix + seq

            seq = seq[:seq_len]

            use_len = len(seq)

            full_text = {}
            full_text["input_ids"] = seq + [0] * (seq_len - use_len)
            full_text["attention_mask"] = use_len * [1] + [0] * (seq_len - use_len)
            full_text["label"] = seq + [IGNORE_TOKEN_ID] * (seq_len - use_len)
            full_text["labels"] = seq + [IGNORE_TOKEN_ID] * (seq_len - use_len)

            self.data.append(full_text)

    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index]


def train():

    # This serves for single-gpu qlora.
    if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1))==1:
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
        if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are incompatible with QLoRA."
            )

    is_chat_model = 'chat' in model_args.model_name_or_path.lower()

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    model = transformers.AutoModelForCausalLM.from_pretrained(
    model_args.model_name_or_path, 
    device_map="cuda", 
    torch_dtype="auto", 
    trust_remote_code=True, 
)
    # model = PeftModel.from_pretrained(model, "/data/Phi-3-mini-4k-instruct/双层lora/checkpoint-383")
    # model.merge_and_unload()
    # model = PeftModel.from_pretrained(model, "/data/Phi-3-mini-4k-instruct/双层lora/checkpoint-84")
    # model.merge_and_unload()
    # model = PeftModel.from_pretrained(model, "/data/Phi-3-mini-4k-instruct/双层lora/checkpoint-2")
    # model.merge_and_unload()
    if model_args.model_checkpoint_bin:
        state_dict = safetensors.torch.load_file(model_args.model_checkpoint_bin)
        model.load_state_dict(state_dict)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        model_max_length=training_args.model_max_length,
        # padding_side="right",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.unk_token  # use unk rather than eos token to prevent endless generation
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    tokenizer.padding_side = 'right'
    # tokenizer.pad_token_id = tokenizer.eod_id

    if training_args.use_lora:
        if lora_args.q_lora or is_chat_model:
            lora_args.lora_target_modules = None
        # else:
        #     modules_to_save = ["embed_tokens", "lm_head"]
        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
            modules_to_save=lora_args.modules_to_save  # This argument serves for adding new tokens.
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )

        model = get_peft_model(model, lora_config)

        # Print peft trainable params
        model.print_trainable_parameters()

        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

        print_trainable_parameters(model)

    # dataset = load_dataset(os.path.dirname(data_args.data_path), data_files=data_args.data_path, split="train")# train[:15%]

    # dataset = dataset.shuffle(42)
    train_dataset = pret_dataset(
        data_args.data_path, tokenizer, training_args.model_max_length
    )
    # print(f'data------------------{len(train_dataset)}')
    # train_dataset = dataset.map(tokenize_function_all, fn_kwargs={"tokenizer": tokenizer, "max_len":training_args.model_max_length})

    # dataset2=load_dataset(data_args.eval_data_path, split='eval')
    dataset2 = load_dataset(
        os.path.dirname(data_args.eval_data_path),
        data_files=data_args.eval_data_path,
        split="train",
    )

    eval_dataset = dataset2.map(
        tokenize_function_all,
        fn_kwargs={"tokenizer": tokenizer, "max_len": training_args.model_max_length},
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    if training_args.evaluate_before_train:
        print(trainer.evaluate())
    trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)

if __name__ == "__main__":
    train()
