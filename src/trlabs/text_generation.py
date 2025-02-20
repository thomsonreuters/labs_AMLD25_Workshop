from typing import List, Optional
import torch
from transformers import PreTrainedTokenizer, PreTrainedModel, TextStreamer
from logging import getLogger

logger = getLogger(__name__)

def generate_text(
    prompt: str,
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    config: Optional['GenerationConfig'] = None,
    device: str = "cuda",
    enable_streaming: bool = False,
    system_message: Optional[str] = None,
) -> List[str]:
    """
    Generate text based on a given prompt using a pre-trained model.
    Args:
        prompt: Input text prompt for generation
        tokenizer: Hugging Face tokenizer instance
        model: Hugging Face model instance
        config: Generation configuration parameters
        device: Computing device ("cuda" or "cpu")
        enable_streaming: Whether to enable text streaming
        system_message: Optional system message to prepend to the conversation
    Returns:
        List of generated text responses
    """
    if not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available")
    
    config = config or GenerationConfig()
    
    try:
        # Prepare messages format
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})
        
        # Tokenize input
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )
        
        # Rest of the code remains the same...
        attention_mask = torch.ones_like(inputs)
        inputs = inputs.to(device)
        attention_mask = attention_mask.to(device)
        
        streamer = TextStreamer(tokenizer) if enable_streaming else None
        
        response = model.generate(
            input_ids=inputs,
            attention_mask=attention_mask,
            streamer=streamer,
            max_new_tokens=config.max_new_tokens,
            use_cache=config.use_cache,
            temperature=config.temperature,
            do_sample=config.do_sample,
        )
        
        generated_texts = [
            tokenizer.decode(
                resp[inputs.shape[1]:],
                skip_special_tokens=True,
            )
            for resp in response
        ]
        return generated_texts
        
    except Exception as e:
        logger.error(f"Error during text generation: {str(e)}")
        raise

class GenerationConfig:
    """Configuration for text generation parameters."""
    def __init__(
        self,
        max_new_tokens: int = 1024,
        temperature: float = 0.0,
        use_cache: bool = True,
        do_sample: bool = False
    ):
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_cache = use_cache
        self.do_sample = do_sample
