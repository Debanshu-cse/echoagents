"""
Module to generate responses using a Hugging Face model with RAG support.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PhiForCausalLM
import importlib.util
import os
import json
from utils import logger
from config import settings
from core import rag

# Global variables to store model and tokenizer
model = None
tokenizer = None
use_rag = True  # Set to False to disable RAG

def is_accelerate_available():
    """Check if the accelerate package is available."""
    return importlib.util.find_spec("accelerate") is not None

def ensure_model_type_in_config(config_path, original_model_name):
    """Ensure the config.json contains a proper model_type."""
    if not os.path.exists(config_path):
        return False
        
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        # Check if model_type is missing
        if 'model_type' not in config:
            # Get model_type from original model
            original_config_path = os.path.join(current_dir, 'original_config.json')
            if os.path.exists(original_config_path):
                with open(original_config_path, 'r') as f:
                    original_config = json.load(f)
                model_type = original_config.get('model_type', 'phi')
            else:
                # Default to 'phi' for phi-2 models
                model_type = 'phi'
                
            # Add model_type to config
            config['model_type'] = model_type
            
            # Write updated config back
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Added missing model_type '{model_type}' to config.json")
            return True
    except Exception as e:
        logger.error(f"Error updating config.json: {e}")
        
    return False

def load_model():
    """
    Load the Hugging Face model, tokenizer, and RAG system.
    """
    global model, tokenizer, use_rag
    
    # Initialize RAG system if enabled
    if use_rag:
        logger.info("Initializing RAG system...")
        rag.initialize_rag()
    
    if settings.USE_CUSTOM_MODEL:
        logger.info(f"Loading custom model from: {settings.CUSTOM_MODEL_PATH}")
        try:
            # First check if the model and tokenizer directories exist
            if not os.path.exists(settings.CUSTOM_MODEL_PATH):
                raise ValueError(f"Custom model directory not found: {settings.CUSTOM_MODEL_PATH}")
            if not os.path.exists(settings.CUSTOM_TOKENIZER_PATH):
                raise ValueError(f"Custom tokenizer directory not found: {settings.CUSTOM_TOKENIZER_PATH}")
            
            # Load the tokenizer
            logger.info(f"Loading custom tokenizer from: {settings.CUSTOM_TOKENIZER_PATH}")
            tokenizer = AutoTokenizer.from_pretrained(
                settings.CUSTOM_TOKENIZER_PATH,
                trust_remote_code=True
            )
            
            # Set pad_token to eos_token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                logger.info("Set pad_token to eos_token for tokenizer")
            
            # Load the model with additional trust_remote_code=True option
            logger.info(f"Loading custom model with trust_remote_code=True")
            model_kwargs = {"torch_dtype": torch.float16}
            
            if is_accelerate_available():
                model_kwargs["device_map"] = "auto"
                logger.info("Using accelerate for model loading with device_map")
                
            model = AutoModelForCausalLM.from_pretrained(
                settings.CUSTOM_MODEL_PATH,
                trust_remote_code=True,
                **model_kwargs
            )
            
            logger.info("Custom model loaded successfully")
            
        except Exception as e:
            logger.critical(f"Failed to load custom model: {e}")
            logger.warning("Falling back to default model")
            load_default_model()
    else:
        load_default_model()

def load_default_model():
    """Load the default Hugging Face model."""
    global model, tokenizer
    
    logger.info(f"Loading model: {settings.MODEL_NAME}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        
        # Set pad_token to eos_token to fix padding issue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info("Set pad_token to eos_token for tokenizer")
        
        # Configure model loading based on available packages
        model_kwargs = {"torch_dtype": torch.float16}
        
        if is_accelerate_available():
            model_kwargs["device_map"] = "auto"
            logger.info("Using accelerate for model loading with device_map")
        else:
            logger.warning("Package 'accelerate' not found; using CPU only. Install with 'pip install accelerate'")
            
        model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_NAME,
            **model_kwargs
        )
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to load model: {e}")
        raise

def generate_response(prompt):
    """
    Generate a response using the loaded Hugging Face model with RAG augmentation.
    
    Args:
        prompt (str): The user's query
    
    Returns:
        str: The model's response
    """
    global model, tokenizer, use_rag
    
    # Load the model if not loaded
    if model is None or tokenizer is None:
        load_model()
    
    try:
        # Augment prompt with RAG context if enabled
        if use_rag:
            augmented_prompt, context = rag.augment_prompt(prompt, num_context_docs=3)
            logger.info(f"Using RAG-augmented prompt with {len(context)} context documents")
        else:
            augmented_prompt = prompt
        
        # Format input for the model
        input_text = f"User: {augmented_prompt}\nAssistant:"
        
        # Tokenize the input without padding to avoid issues
        inputs = tokenizer(input_text, return_tensors="pt")
        
        # Move inputs to the appropriate device if model is on a specific device
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        logger.info("Generating response...")
        with torch.no_grad():
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs.get("attention_mask", None),
                max_length=settings.MAX_LENGTH + len(inputs["input_ids"][0]),
                temperature=settings.TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Decode and format response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response - improved parsing
        if "Assistant:" in full_response:
            response = full_response.split("Assistant:", 1)[1].strip()
        else:
            # Get the response after the input prompt
            response = full_response[len(input_text):].strip()
            if not response:  # Fallback if parsing fails
                response = full_response.strip()
        
        logger.info(f"Response generated: {response[:50]}...")
        return response
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return "Sorry, I couldn't process that request."
