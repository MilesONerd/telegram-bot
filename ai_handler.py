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
            'llama': {
                'name': 'nvidia/Llama-3.1-Nemotron-70B-Instruct-HF',
                'type': 'causal',
                'task': 'text-generation'
            },
            'bart': {
                'name': 'facebook/bart-large',
                'type': 'conditional',
                'task': 'summarization'
            }
        }
        
        self.default_model = os.getenv('DEFAULT_MODEL', 'llama')
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
            
            # Initialize Llama for general text generation
            logger.info(f"Loading Llama model: {self.model_configs['llama']['name']}")
            try:
                self.tokenizers['llama'] = AutoTokenizer.from_pretrained(
                    self.model_configs['llama']['name'],
                    local_files_only=False
                )
                logger.info("Llama tokenizer loaded successfully")
                
                self.models['llama'] = AutoModelForCausalLM.from_pretrained(
                    self.model_configs['llama']['name'],
                    device_map='auto' if torch.cuda.is_available() else None,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    local_files_only=False
                )
                logger.info("Llama model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Llama model: {str(e)}")
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
        max_length: int = 300,
        temperature: float = 0.2,
        top_p: float = 0.4,
        max_attempts: int = 5
    ) -> str:
        """
        Generate a response using the specified model.
        
        Args:
            text: Input text to generate response from
            model_key: Key of the model to use (default: None, uses default_model)
            max_length: Maximum length of generated text
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            max_attempts: Maximum number of attempts to generate a valid response
            
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
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(model.device)
            
            # Attempts to generate a response
            attempts = 0
            response = ""
            
            while attempts < max_attempts:
                attempts += 1
                
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
                
                # Decode response
                response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
                
                # Check that the answer is not repetitive or meaningless
                if len(response.split()) > 3 and response != text:  # Check if the response is valid
                    break  # If the response is valid, exit the loop
                
            if attempts == max_attempts and (not response or response == text):
                return "Sorry, I could not generate a meaningful response after multiple attempts."
        
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
