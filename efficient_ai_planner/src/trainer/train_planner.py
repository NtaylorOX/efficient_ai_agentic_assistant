# --- SFTTrainer Example for CoT Planner Data ---

# 1. Import necessary libraries
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer # SFT specific
import torch
import argparse
import os

# --- Main Function ---
def main(args):
    # 2. Configuration
    base_model_id = args.base_model_id
    dataset_path = args.dataset_path # Path to your generated .jsonl file (CoT format)
    output_dir = args.output_dir

    print("--- Configuration ---")
    print(f"Base Model ID: {base_model_id}")
    print(f"Dataset Path (Expecting CoT format): {dataset_path}")
    print(f"Output Directory: {output_dir}")
    print("---------------------")

    # 3. Load Dataset
    print(f"\nLoading dataset from: {dataset_path}")
    # Use the 'json' loading script for .jsonl files
    try:
        # Ensure the dataset has 'problem_description' and 'gemini_cot_reasoning' columns
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        print(f"Dataset loaded successfully. Number of records: {len(dataset)}")
        # Verify expected columns exist
        required_columns = ['problem_description', 'gemini_cot_reasoning']
        if not all(col in dataset.column_names for col in required_columns):
             print(f"Error: Dataset missing required columns. Expected: {required_columns}, Found: {dataset.column_names}")
             return
        print("Example record:")
        print(dataset[0]) # Print first record to verify structure
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 4. Load Tokenizer
    print(f"\nLoading tokenizer for: {base_model_id}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    # Set padding token if missing
    if tokenizer.pad_token is None:
        print("Setting pad_token to eos_token")
        tokenizer.pad_token = tokenizer.eos_token
    # SFT often uses right padding, but check model specifics
    tokenizer.padding_side = "right"
    print("Tokenizer loaded.")

    # 5. Load Model with Quantization (QLoRA)
    print("\nLoading base model with QLoRA...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        device_map="auto", # Automatically distribute across available GPUs
        attn_implementation="flash_attention_2" # Use Flash Attention if available
    )
    model.config.use_cache = False # Important for training stability
    # Optional: Set pretraining_tp if needed for some models like Llama
    # model.config.pretraining_tp = 1
    print("Base model loaded.")

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    print("Model prepared for k-bit training.")

    # 6. LoRA Configuration
    print("\nConfiguring LoRA...")
    peft_config = LoraConfig(
        r=16,                 # Rank of the LoRA matrices
        lora_alpha=32,        # Alpha scaling factor
        lora_dropout=0.05,    # Dropout probability
        bias="none",          # Bias type ('none', 'all', or 'lora_only')
        task_type="CAUSAL_LM", # Task type
        # Target modules vary by model architecture. Common ones for Llama/Mistral:
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    print("LoRA configured.")

    # 7. SFT Training Arguments
    print("\nConfiguring Training Arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,           # Number of training epochs
        per_device_train_batch_size=args.batch_size, # Batch size per GPU
        gradient_accumulation_steps=args.gradient_accumulation, # Accumulate gradients
        optim="paged_adamw_32bit", # Optimizer suitable for QLoRA
        save_strategy="epoch",      # Save checkpoint every epoch
        logging_steps=20,           # Log training metrics every N steps
        learning_rate=2e-4,         # Learning rate (can be higher for SFT than DPO)
        weight_decay=0.001,         # Weight decay
        fp16=False,                 # Use mixed precision (choose one)
        bf16=True,                  # Use bfloat16 for better performance on modern GPUs
        max_grad_norm=0.3,          # Gradient clipping
        warmup_ratio=0.03,          # Warmup ratio
        lr_scheduler_type="constant",# Learning rate scheduler
        # group_by_length=True,     # Group sequences of similar length (speeds up training)
        report_to="wandb",          # Optional: report metrics to Weights & Biases
        # gradient_checkpointing=True, # Enable gradient checkpointing to save memory
        # gradient_checkpointing_kwargs={'use_reentrant':False} # Recommended for newer PyTorch versions
    )
    print("Training Arguments configured.")

    # --- Data Formatting for CoT Data ---
    # This function formats the CoT data into a single text sequence
    # suitable for instruction fine-tuning.
    def format_cot_instruction(sample):
        """
        Formats a sample from the CoT dataset for SFT training.

        Args:
            sample (dict): A dictionary containing 'problem_description'
                           and 'gemini_cot_reasoning'.

        Returns:
            dict: A dictionary with a 'text' key containing the formatted string.
        """
        instruction = sample['problem_description']
        response = sample['gemini_cot_reasoning']

        # Choose a formatting style appropriate for the base model.
        # Example 1: Generic Instruction format
        # formatted_text = f"### Instruction:\n{instruction}\n\n### Response:\n{response}"

        # Example 2: Llama 3 Instruct format (using chat template is preferred)
        # Note: Applying chat templates usually requires tokenization first or
        # careful string construction. This is a simplified string version.
        # For production, use tokenizer.apply_chat_template if possible.
        formatted_text = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{response}<|eot_id|>"


        # Ensure EOS token is present if not added by the template/format
        # (Llama 3 template above includes <|eot_id|>, which acts as EOS)
        # if not formatted_text.endswith(tokenizer.eos_token):
        #      formatted_text += tokenizer.eos_token

        return {"text": formatted_text} # SFTTrainer default text column is 'text'

    print("\nFormatting dataset (CoT format)...")
    # Apply the formatting function to the dataset
    formatted_dataset = dataset.map(format_cot_instruction)
    print("Dataset formatted.")
    print("Example formatted text:")
    # Ensure the first record has the required keys before printing
    if 'text' in formatted_dataset.column_names and len(formatted_dataset) > 0:
         print(formatted_dataset[0]['text'])
    else:
         print("Could not display example formatted text (column 'text' not found or dataset empty).")


    # 8. Initialize SFTTrainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset, # Use the formatted dataset
        dataset_text_field="text",   # Specify the column containing formatted text
        peft_config=peft_config,     # Pass LoRA config here
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length, # Max sequence length for packing/truncation
        packing=True,                # Pack multiple short sequences into one (efficient)
        # formatting_func=format_cot_instruction, # Can pass func here instead of mapping
    )
    print("SFTTrainer initialized.")

    # 9. Start SFT Training
    print("\n--- Starting Training ---")
    trainer.train()
    print("--- Training Finished ---")

    # 10. Save the fine-tuned SFT adapter
    print(f"\nSaving model adapter to {output_dir}")
    trainer.save_model(output_dir) # Saves the LoRA adapter
    # tokenizer.save_pretrained(output_dir) # Optionally save tokenizer too
    print(f"Model adapter saved successfully to {output_dir}")

# --- Argument Parser ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model using SFTTrainer on CoT planner data.")

    parser.add_argument("--base_model_id", type=str, required=True, help="Hugging Face model ID or path to the base model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the .jsonl dataset file (CoT format: {'problem_description': ..., 'gemini_cot_reasoning': ...}).")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned LoRA adapter.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size (CoT data can be long, smaller batch often needed).")
    parser.add_argument("--gradient_accumulation", type=int, default=8, help="Gradient accumulation steps (increase if using small batch size).")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length for training (adjust based on CoT data length and model capacity).")

    # Example Usage:
    # python train_sft_cot_planner.py \
    #   --base_model_id "meta-llama/Llama-3.1-8B-Instruct" \
    #   --dataset_path ./synthetic_training_data/gemini_cot_plans.jsonl \
    #   --output_dir ./sft_cot_planner_adapter \
    #   --epochs 1 \
    #   --batch_size 1 \
    #   --gradient_accumulation 8 \
    #   --max_seq_length 2048

    args = parser.parse_args()
    main(args)
