# --- Inference Script for Fine-tuned CoT Planner ---

# 1. Import necessary libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse
import os

# --- Main Inference Function ---
def run_inference(args):
    # 2. Configuration
    base_model_id = args.base_model_id
    adapter_path = args.adapter_path # Path to your saved LoRA adapter
    prompt_text = args.prompt

    print("--- Configuration ---")
    print(f"Base Model ID: {base_model_id}")
    print(f"Adapter Path: {adapter_path}")
    print("---------------------")

    # 3. Load Tokenizer (Same as training)
    print(f"\nLoading tokenizer for: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    # Ensure padding token is set (often needed for batching, though less critical for single inference)
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    # Padding side doesn't matter as much for single generation unless batching
    tokenizer.padding_side = "left" # Often recommended for generation
    print("Tokenizer loaded.")

    # 4. Load Base Model with Quantization (QLoRA)
    # Use the same quantization config as during training
    print("\nLoading base model with QLoRA (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True, # Match training config
    )

    # Load base model quantized
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto", # Load onto available GPU(s)
        attn_implementation="flash_attention_2" # Use if available
    )
    # No need for prepare_model_for_kbit_training during inference
    print("Base model loaded.")

    # 5. Load LoRA Adapter
    print(f"\nLoading LoRA adapter from: {adapter_path}")
    # Automatically uses the device map from the base model
    model = PeftModel.from_pretrained(model, adapter_path)
    print("LoRA adapter loaded.")
    # Optional: Merge adapter for potentially faster inference if memory allows
    # print("\nMerging adapter...")
    # model = model.merge_and_unload()
    # print("Adapter merged.")

    # Set model to evaluation mode
    model.eval()

    # 6. Prepare the Prompt
    print("\n--- Preparing Prompt ---")
    # IMPORTANT: Use the EXACT same formatting as used during training!
    # Example using the Llama 3 Instruct format from the training script:
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    print(f"Formatted Prompt:\n{formatted_prompt}")

    # Tokenize the input
    inputs = tokenizer(formatted_prompt, return_tensors="pt", padding=False).to(model.device) # Move inputs to GPU

    # 7. Generate Response
    print("\n--- Generating Response ---")
    # Set generation parameters
    generation_params = {
        "max_new_tokens": args.max_new_tokens, # Control output length
        "temperature": args.temperature,   # Control randomness (lower for more deterministic)
        "top_p": args.top_p,           # Nucleus sampling
        "do_sample": True,         # Enable sampling (needed for temp/top_p)
        "pad_token_id": tokenizer.eos_token_id # Set pad token ID for generation
    }
    print(f"Generation parameters: {generation_params}")

    with torch.no_grad(): # Disable gradient calculations for inference
        outputs = model.generate(**inputs, **generation_params)

    # Decode the generated tokens, skipping special tokens and the prompt
    # We need to decode only the newly generated tokens
    input_length = inputs.input_ids.shape[1]
    generated_ids = outputs[0, input_length:]
    response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

    print("\n--- Generated CoT Reasoning/Plan ---")
    print(response_text)
    print("-------------------------------------")


# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a fine-tuned LoRA model for CoT planning.")

    parser.add_argument("--base_model_id", type=str, required=True, help="Hugging Face model ID or path to the base model.")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to the directory containing the saved LoRA adapter weights.")
    parser.add_argument("--prompt", type=str, required=True, help="The problem description prompt text.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature for generation.")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling probability.")

    # Example Usage:
    # python run_inference_cot.py \
    #   --base_model_id "meta-llama/Llama-3.1-8B-Instruct" \
    #   --adapter_path ./sft_cot_planner_adapter \
    #   --prompt "You are visiting City X... [Your full problem description here]" \
    #   --max_new_tokens 1500 \
    #   --temperature 0.5

    args = parser.parse_args()
    run_inference(args)
