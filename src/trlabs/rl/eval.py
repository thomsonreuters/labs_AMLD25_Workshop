import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, AutoPeftModelForCausalLM

def setup_model_and_lora(
    # Base model path
    base_model_name, 
    # LoRA adapter path
    lora_path = None
):
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    
    
    if lora_path is not None:
        # Load and merge LoRA weights
        model = AutoPeftModelForCausalLM.from_pretrained(
            lora_path,
            device_map="auto",
            torch_dtype=torch.float16
        )
        model = model.merge_and_unload()  # Merge LoRA weights into base model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    model.eval()
    
    return model, tokenizer

# Generate text
def generate(prompt, model, tokenizer, max_length = 7688):
    inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True).to(model.device)
    
    outputs = model.generate(
        input_ids = inputs,
        max_length = max_length,
        do_sample = False,
        top_k = None,
        top_p = None,
        temperature = None
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)