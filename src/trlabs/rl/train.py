import torch

from typing import Optional, List, Dict
import datasets
from datasets import (
    load_dataset, 
    load_from_disk, 
    DatasetDict,
    concatenate_datasets
)

from accelerate import Accelerator, PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer


from trl import (
    ModelConfig,
    DPOTrainer,
    DPOConfig,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE

from .data import *


def dpo(
    data_params: dict, 
    training_params: dict, 
    model_config: dict 
):
    accelerator = Accelerator()
    
    
    data_args = DataArguments(**data_params)
    training_args =  DPOConfig(**training_params)
    model_args = ModelConfig(**model_config)
    
    ###################
    # Model & Tokenizer
    ###################
    torch_dtype = (
        model_args.torch_dtype
        if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
    )
    peft_config = get_peft_config(model_args)
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    
    ################
    # Dataset
    ################
    dataset =  get_datasets(data_args, splits=data_args.dataset_splits)
    
    
    
    ################
    # Training
    ################
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[data_args.dataset_train_split],
        eval_dataset=dataset[data_args.dataset_test_split].shuffle(data_args.seed)\
            .take(min(len(dataset[data_args.dataset_test_split]), data_args.num_eval_samples))\
            if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    
    trainer.train()
    
    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
    
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=data_args.dataset_name)