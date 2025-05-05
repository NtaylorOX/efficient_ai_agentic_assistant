# --- SFTTrainer Example for Step-by-Step Planner ---

# 1. Import necessary libraries
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM # SFT specific
import torch
import argparse
import os

# --- Main Function ---
def main(args):
    # 2. Configuration
    base_model_id = args.base_model_id
    dataset_path = args.dataset_path # Path to your generated .jsonl file
    output_dir = args.output_dir

    print("--- Configuration ---")
    print(f"Base Model ID: {base_model_id}")
    print(f"Dataset Path: {dataset_path}")
    print(f"Output Directory: {output_dir}")
    print("---------------------")

    # 3. Load Dataset
    print(f"\nLoading dataset from: {dataset_path}")
    # Use the 'json' loading script for .jsonl files
    try:
        # Ensure the dataset has 'prompt' and 'completion' columns as generated
        dataset = load_dataset('json', data_files=dataset_path, split='train')
        # Rename 'completion' to 'response' if needed by collator, or format later
        # dataset = dataset.rename_column("completion", "response")
        print(f"Dataset loaded successfully. Number of steps: {len(dataset)}")
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

    # Apply PEFT to the model ( SFTTrainer can also handle this if passed directly)
    # model = get_peft_model(model, peft_config)
    # model.print_trainable_parameters()
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

    # Optional: Format dataset for SFTTrainer
    # SFTTrainer expects a text column or a specific format.
    # Let's format prompt and completion into a single text sequence.
    # Many models expect specific chat templates or instruction formats.
    # Example formatting function (adjust based on target model's expected format):
    def format_instruction(sample):
        # Simple formatting: combine prompt and completion
        # Ideally, use the tokenizer's chat template if available and appropriate
        # e.g., tokenizer.apply_chat_template([{'role':'user', 'content': sample['prompt']}, {'role':'assistant', 'content': sample['completion']}], tokenize=False)
        formatted_text = f"### Instruction:\n{sample['prompt']}\n\n### Response:\n{sample['completion']}"
        # Add EOS token if tokenizer doesn't automatically
        if not formatted_text.endswith(tokenizer.eos_token):
             formatted_text += tokenizer.eos_token
        return {"text": formatted_text} # SFTTrainer default text column is 'text'

    print("\nFormatting dataset...")
    formatted_dataset = dataset.map(format_instruction)
    print("Dataset formatted.")
    print("Example formatted text:")
    print(formatted_dataset[0]['text'])


    # 8. Initialize SFTTrainer
    print("\nInitializing SFTTrainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=formatted_dataset, # Use the formatted dataset
        # dataset_text_field="text",   # Specify the column containing formatted text
        peft_config=peft_config,     # Pass LoRA config here
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length, # Max sequence length for packing/truncation
        packing=True,                # Pack multiple short sequences into one (efficient)
        # formatting_func=format_instruction, # Alternatively, pass formatting func here
        # data_collator=...          # Optional: Use a custom collator if needed
        # Example: Use completion-only collator if you formatted like chat
        # response_template = "### Response:"
        # collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
        # data_collator=collator,
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
    parser = argparse.ArgumentParser(description="Fine-tune a model using SFTTrainer on step-by-step planner data.")

    parser.add_argument("--base_model_id", type=str, required=True, help="Hugging Face model ID or path to the base model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the .jsonl dataset file generated by synthetic_planner_data_generator.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned LoRA adapter.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=2, help="Per-device training batch size.")
    parser.add_argument("--gradient_accumulation", type=int, default=4, help="Gradient accumulation steps.")
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Maximum sequence length for training.")

    # Example Usage:
    # python train_sft_planner.py --base_model_id "meta-llama/Llama-3.1-8B-Instruct" --dataset_path ./synthetic_training_data/planner_steps.jsonl --output_dir ./sft_planner_adapter --epochs 1 --batch_size 2 --gradient_accumulation 4 --max_seq_length 1024

    args = parser.parse_args()
    main(args)
