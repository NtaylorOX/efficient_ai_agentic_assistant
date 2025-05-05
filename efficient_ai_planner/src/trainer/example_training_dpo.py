# --- DPOTrainer Example ---

# 1. Import necessary libraries
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from trl import DPOTrainer, DPOConfig
import torch

# 2. Configuration
# Use the SFT-tuned model adapter path or the base model if no SFT was done
model_id_or_path = "./sft_finetuned_model" # Path to the SFT adapter OR base model ID
# If using SFT adapter, specify the base model it was trained on
base_model_id = "meta-llama/Llama-3.1-8B-Instruct" # Or "microsoft/Phi-4-mini-4k-instruct"

# Preference dataset (must contain 'prompt', 'chosen', 'rejected' columns)
# Example dataset: "trl-lib/ultrafeedback_binarized" or a custom one
dataset_name = "trl-lib/ultrafeedback_binarized"
output_dir = "./dpo_finetuned_model"

# 3. Load Preference Dataset
# Ensure dataset has 'prompt', 'chosen', 'rejected' columns
# The 'chosen' and 'rejected' columns should contain the full response text
dataset = load_dataset(dataset_name, split="train_prefs") # Adjust split name as needed

# Optional: Pre-process dataset if needed (e.g., apply chat template)
# The DPOTrainer can often handle formatting if the tokenizer has a chat template

# 4. Load Tokenizer (same as SFT)
tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left" # DPO often uses left padding

# 5. Load Model with Quantization (QLoRA)
# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# Load the base model first
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation="flash_attention_2"
)
model.config.use_cache = False

# Load the SFT adapter if applicable
# If model_id_or_path points to an adapter, load it like this:
# model = PeftModel.from_pretrained(model, model_id_or_path)
# Or, if loading adapter separately:
# model = AutoPeftModelForCausalLM.from_pretrained(
#     model_id_or_path, # Path to SFT adapter
#     low_cpu_mem_usage=True,
#     torch_dtype=torch.float16, # Or bfloat16
#     load_in_4bit=True, # Reload with quantization if needed
#     is_trainable=True
# )

# Prepare model for k-bit training if loading base model + adapter separately
model = prepare_model_for_kbit_training(model)

# 6. LoRA Configuration (can be the same as SFT or different)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)

# Apply PEFT to the model
model = get_peft_model(model, peft_config)

# 7. DPO Configuration (inherits from TrainingArguments)
# Use DPOConfig for DPO-specific parameters like beta
training_args = DPOConfig(
    output_dir=output_dir,
    num_train_epochs=1, # DPO often requires fewer epochs
    per_device_train_batch_size=2, # DPO often uses smaller batch sizes
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=5e-5, # DPO often uses a lower learning rate than SFT
    weight_decay=0.001,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="constant",
    max_prompt_length=512, # Max length of the prompt section
    max_length=1024, # Max length of prompt + completion
    beta=0.1, # The beta parameter in DPO loss, controls divergence from reference model
    # loss_type="sigmoid", # Default DPO loss type
    report_to="wandb", # Optional
    # gradient_checkpointing=True, # Can save memory
    # gradient_checkpointing_kwargs={'use_reentrant':False}
)

# 8. Initialize DPOTrainer
# DPOTrainer needs the main model and optionally a reference model.
# If ref_model is None, DPOTrainer creates one internally (often needed for PEFT).
dpo_trainer = DPOTrainer(
    model=model,
    ref_model=None, # Set to None when using PEFT, trainer handles it
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer,
    peft_config=peft_config, # Pass PEFT config
    # data_collator=... # Optional
    # eval_dataset=... # Optional
)

# 9. Start DPO Training
dpo_trainer.train()

# 10. Save the fine-tuned DPO adapter
dpo_trainer.save_model(output_dir)
print(f"Model adapter saved to {output_dir}")

# Optional: Merge adapter and save full model
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained(f"{output_dir}/final_merged_model")
# tokenizer.save_pretrained(f"{output_dir}/final_merged_model")