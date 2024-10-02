
"""
LLM Models Configuration
"""

from transformers import BitsAndBytesConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from templates import *

ACCESS_TOKEN = "" # INSERT HF API TOKEN HERE

def bytes_to_gb(b):
    return b / 1024 / 1024 / 1024

class ModelConfig:
    # name, model, tokenizer, and template
    def __init__(self, name):
        self.name = name
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, token=ACCESS_TOKEN, padding_side='left')

    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def get_model_size(self):
        return bytes_to_gb(self.model.get_memory_footprint())
    
    def set_bos_token(self):
        self.tokenizer.bos_token = self.tokenizer.eos_token

    def set_pad_token(self):
        self.tokenizer.pad_token = self.tokenizer.eos_token  

    def add_calculator_tokens(self):
        special_tokens = ['||STARTWORK||', '||ENDWORK||', '||ENDCALC||']
        self.tokenizer.add_tokens(special_tokens)
        self._resize_token_embeddings()     

    def load_peft_model(self, save_dir):
        print(f"Loading PEFT model from {save_dir}")
        self.tokenizer = AutoTokenizer.from_pretrained(save_dir, padding_side='left')
        self._resize_token_embeddings()
        self.model = PeftModel.from_pretrained(self.model, save_dir, is_trainable=True)

        assert self.model.get_input_embeddings().weight.size(0) == len(self.tokenizer)
        print(f"Model loaded. Device: {self.model.device}")
        
    def _resize_token_embeddings(self):
        self.model.resize_token_embeddings(len(self.tokenizer))

class DeepSeek7BConfig(ModelConfig):
    """ 
    DeepSeek-7B Model Configuration. 
    """
    def __init__(self):
        super().__init__("deepseek-ai/deepseek-math-7b-rl")
        config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto', quantization_config=config)
        self.tokenizer.chat_template = DEEPSEEK_TEMPLATE

        self.set_pad_token()

class Falcon7BConfig(ModelConfig):
    """ 
    Falcon-7B Model Configuration. 
    Contains eos_token. No bos_token, unk_token, pad_token.
    """
    def __init__(self):
        super().__init__("tiiuae/falcon-7b")
        config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto', quantization_config=config)
        self.tokenizer.chat_template = FALCON_TEMPLATE

        self.set_bos_token()
        self.set_pad_token()

class Gemma2Config(ModelConfig):
    """ 
    Gemma-2 Model Configuration. 
    Contains bos_token, eos_token, unk_token, pad_token.
    """
    def __init__(self):
        super().__init__("google/gemma-2b")
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto')
        self.tokenizer.chat_template = GEMMA_TEMPLATE

class Llama3_8BConfig(ModelConfig):
    """ 
    Llama3-8B Model Configuration. 
    Contains bos_token, eos_token. No pad_token.
    """
    def __init__(self):
        super().__init__("meta-llama/Meta-Llama-3-8B-Instruct")
        config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto', quantization_config=config)
        # Already has in-built chat template

        self.set_pad_token()

class Llama3a_8BConfig(ModelConfig):
    """ 
    Llama3.1-8B Model Configuration. 
    Contains bos_token, eos_token. No pad_token.
    """
    def __init__(self):
        super().__init__("meta-llama/Meta-Llama-3.1-8B-Instruct")
        config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto', quantization_config=config)
        # Already has in-built chat template

        self.set_pad_token()

class Mathstral7BConfig(ModelConfig):
    """ 
    Mathstral-7B Model Configuration. 
    """
    def __init__(self):
        super().__init__("mistralai/mathstral-7B-v0.1")
        config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto', quantization_config=config)
        self.tokenizer.chat_template = MISTRAL_TEMPLATE

        self.set_pad_token()

class MetaMathMistral7BConfig(ModelConfig):
    """ 
    MetaMathLlemma-7B Model Configuration. 
    Contains bos_token, eos_token, pad_token, unk_token.
    """
    def __init__(self):
        super().__init__("meta-math/MetaMath-Mistral-7B")
        config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto', quantization_config=config)
        self.tokenizer.chat_template = METAMATH_TEMPLATE
    
class Mistral7BConfig(ModelConfig):
    """ 
    Mistral-7B Model Configuration. 
    Contains bos_token, eos_token, unk_token. No pad_token.
    """
    def __init__(self):
        super().__init__("mistralai/Mistral-7B-v0.3")
        config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto', quantization_config=config)
        self.tokenizer.chat_template = MISTRAL_TEMPLATE

        self.set_pad_token()

class Phi2Config(ModelConfig):
    """ 
    Phi-2 Model Configuration. 
    Contains bos_token, eos_token, unk_token. No pad_token.
    """
    def __init__(self):
        super().__init__("microsoft/phi-2")
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto')
        self.tokenizer.chat_template = PHI2_TEMPLATE

        self.set_pad_token()

class Rho1BConfig(ModelConfig):
    """ 
    Rho-1B Model Configuration. 
    """
    def __init__(self):
        super().__init__("realtreetune/rho-1b-sft-MATH")
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto')
        self.tokenizer.chat_template = TINYLLAMA_TEMPLATE

        self.set_pad_token()

class TinyLlamaConfig(ModelConfig):
    """ 
    TinyLlama (1.1B) Model Configuration. 
    Contains bos_token, eos_token, pad_token, unk_token.
    """
    def __init__(self):
        super().__init__("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto')
        
    
class Qwen7BConfig(ModelConfig):
    """
    Qwen-7B Model Configuration.
    Contains eos_token, pad_token
    """
    def __init__(self):
        super().__init__("Qwen/Qwen2-7B-Instruct")
        config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto', quantization_config=config)
        # Already has in-built chat template

        
class Vicuna7BConfig(ModelConfig):
    """
    Vicuna-7B Model Configuration.
    Contains bos_token, eos_token, unk_token, pad_token
    """
    def __init__(self):
        super().__init__("lmsys/vicuna-7b-v1.5")
        config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto', quantization_config=config)
        self.tokenizer.chat_template = VICUNA_TEMPLATE


class WizardMath7BConfig(ModelConfig):
    """ 
    MetaMathLlemma-7B Model Configuration. 
    Contains bos_token, eos_token, pad_token, unk_token.
    """
    def __init__(self):
        super().__init__("WizardLMTeam/WizardMath-7B-V1.1")
        config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto', quantization_config=config)
        self.tokenizer.chat_template = METAMATH_TEMPLATE
        

class Zephyr3BConfig(ModelConfig):
    """
    Zephyr-3B Model Configuration.
    Contains bos_token, eos_token, unk_token, pad_token
    """
    def __init__(self):
        super().__init__("stabilityai/stablelm-zephyr-3b")
        self.model = AutoModelForCausalLM.from_pretrained(self.name, token=ACCESS_TOKEN, device_map='auto')
        # Already has in-built chat template


MODELS = {
    "deepseekmath-7b": DeepSeek7BConfig,
    "falcon-7b": Falcon7BConfig,
    "gemma-2b": Gemma2Config,
    "llama3-8b": Llama3_8BConfig, 
    "llama3_1-8b": Llama3a_8BConfig,
    "mathstral-7b": Mathstral7BConfig,
    "metamathmistral-7b": MetaMathMistral7BConfig,
    "mistral-7b": Mistral7BConfig,
    "phi-2": Phi2Config,
    "qwen-7b": Qwen7BConfig,
    "rho-1b": Rho1BConfig,
    "tinyllama": TinyLlamaConfig,
    "wizardmath-7b": WizardMath7BConfig, 
    "vicuna-7b": Vicuna7BConfig,
    "zephyr-3b": Zephyr3BConfig
}

