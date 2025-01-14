"""
AI Handler Module for MilesONerd AI Bot
Manages model initialization, text generation, and model interactions
"""
import logging
import os
from typing import Dict, Optional, List, Any
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BartForConditionalGeneration,
    BartTokenizer,
    pipeline
)

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class AIModelHandler:
    """Handles AI model operations including initialization and text generation."""
    
    def __init__(self):
        """Initialize the AI model handler with model configurations."""
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        
        # Model configurations
        self.model_configs = {
            'gpt2': {
                'name': 'openai-community/gpt2',
                'type': 'causal',
                'task': 'text-generation'
            },
            'bart': {
                'name': 'facebook/bart-large',
                'type': 'conditional',
                'task': 'summarization'
            }
        }
        
        self.default_model = os.getenv('DEFAULT_MODEL', 'gpt2')
        self.enable_learning = os.getenv('ENABLE_CONTINUOUS_LEARNING', 'true').lower() == 'true'
        
    async def initialize_models(self) -> bool:
        """
        Initialize AI models asynchronously.
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            logger.info("Starting model initialization...")
            logger.info(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
                logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Initialize BART for summarization
            logger.info(f"Loading BART model: {self.model_configs['bart']['name']}")
            try:
                self.tokenizers['bart'] = BartTokenizer.from_pretrained(
                    self.model_configs['bart']['name'],
                    local_files_only=False
                )
                logger.info("BART tokenizer loaded successfully")
                
                self.models['bart'] = BartForConditionalGeneration.from_pretrained(
                    self.model_configs['bart']['name'],
                    device_map='auto' if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    local_files_only=False
                )
                logger.info("BART model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading BART model: {str(e)}")
                return False
            
            # Initialize GPT-2 for general text generation
            logger.info(f"Loading GPT-2 model: {self.model_configs['gpt2']['name']}")
            try:
                self.tokenizers['gpt2'] = AutoTokenizer.from_pretrained(
                    self.model_configs['gpt2']['name'],
                    local_files_only=False
                )
                logger.info("GPT-2 tokenizer loaded successfully")
                
                self.models['gpt2'] = AutoModelForCausalLM.from_pretrained(
                    self.model_configs['gpt2']['name'],
                    device_map='auto' if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    local_files_only=False
                )
                logger.info("GPT-2 model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading GPT-2 model: {str(e)}")
                return False
            
            logger.info("All models initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing models: {str(e)}")
            return False
    
    async def generate_response(
        self,
        text: str,
        model_key: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 0.3,
        top_p: float = 0.7
    ) -> str:
        """
        Generate a response using the specified model.
        
        Args:
            text: Input text to generate response from
            model_key: Key of the model to use (default: None, uses default_model)
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            
        Returns:
            str: Generated response text
        """
        try:
            model_key = model_key or self.default_model
            
            if model_key not in self.models:
                raise ValueError(f"Model {model_key} not found")
            
            model = self.models[model_key]
            tokenizer = self.tokenizers[model_key]
            
            # Prepare input
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode and return response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return f"I apologize, but I encountered an error while processing your request. Please try again."
    
    async def summarize_text(
        self,
        text: str,
        max_length: int = 130,
        min_length: int = 30
    ) -> str:
        """
        Summarize text using BART model.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            min_length: Minimum length of summary
            
        Returns:
            str: Summarized text
        """
        try:
            inputs = self.tokenizers['bart'](
                text,
                return_tensors="pt",
                truncation=True,
                max_length=1024,
                padding=True
            ).to(self.models['bart'].device)
            
            summary_ids = self.models['bart'].generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                length_penalty=2.0,
                num_beams=4,
                early_stopping=True
            )
            
            summary = self.tokenizers['bart'].decode(
                summary_ids[0],
                skip_special_tokens=True
            )
            return summary.strip()
            
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            return f"I apologize, but I encountered an error while trying to summarize the text. Please try again."
    
    def get_available_models(self) -> List[str]:
        """Get list of available model keys."""
        return list(self.model_configs.keys())
    
    def get_model_info(self, model_key: str) -> Optional[Dict]:
        """Get information about a specific model."""
        return self.model_configs.get(model_key)

# Create singleton instance
ai_handler = AIModelHandler()
