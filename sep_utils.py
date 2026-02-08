"""
Shared utilities for SEP pipeline (Llama-3 Edition).
"""
import logging

def format_llama3_prompt(document):
    """
    Standard Llama-3-Instruct wrapper.
    CRITICAL: This function must be used in both generation and feature extraction.
    """
    system_prompt = "You are a helpful assistant that summarizes articles."
    user_prompt = f"Summarize the following article in one sentence:\n\n{document}"
    
    # Llama 3 explicit special token structure
    # Note: We rely on the tokenizer recognizing these text representations or the model being trained on them.
    # Ideally we would use tokenizer.apply_chat_template, but to keep the wrapper simple string passing:
    formatted = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_prompt}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    return formatted

def setup_simple_logger():
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )