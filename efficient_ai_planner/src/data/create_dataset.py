import vertexai
from vertexai.generative_models import GenerativeModel
from datasets import load_dataset, Dataset, Features, Value, DatasetDict, load_from_disk
import os
import json
import argparse
import time
import ast # Added for safely evaluating string representations if needed later
from dotenv import load_dotenv

# --- Helper Function to Load/Create Dataset ---

def create_dataset_from_custom_json(data_path: str, features: Features = None) -> Dataset | None:
    """
    Loads data from a JSON file structured as a dictionary of records,
    where keys are example IDs and values are the record dictionaries.
    Transforms it into a Hugging Face Dataset.

    Args:
        data_path (str): The path to the input JSON file.
        features (datasets.Features, optional): An optional Features object
            to define the dataset schema explicitly. If None, features will
            be inferred. Defaults to None.

    Returns:
        datasets.Dataset | None: The created Dataset object if successful,
                                 otherwise None.
    """
    print(f"Attempting to load custom JSON data from: {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data_dict = json.load(f)
        print(f"Successfully loaded JSON data with {len(data_dict)} top-level keys.")
    except FileNotFoundError:
        print(f"Error: Input file not found at {data_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {data_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during JSON loading: {e}")
        return None

    print("Transforming data into a list of records...")
    data_list = []
    for example_id, record_data in data_dict.items():
        if isinstance(record_data, dict):
            new_record = record_data.copy()
            # Ensure 'id' field exists, using the original key if not present
            if 'id' not in new_record:
                 new_record['id'] = example_id
            data_list.append(new_record)
        else:
            print(f"Warning: Skipping key '{example_id}' as its value is not a dictionary.")

    if not data_list:
        print("Error: No valid records found after transformation.")
        return None
    print(f"Transformed {len(data_list)} records.")

    print("Creating Hugging Face Dataset object...")
    try:
        if features:
            custom_dataset = Dataset.from_list(data_list, features=features)
            print("Dataset created with provided features.")
        else:
            custom_dataset = Dataset.from_list(data_list)
            print("Dataset created with inferred features.")
        print("\nInferred/Provided Features:")
        print(custom_dataset.features)
        return custom_dataset
    except Exception as e:
        print(f"Error creating Dataset object: {e}")
        return None

# --- Main Script Logic ---

def main(args):
    """
    Main function to load data, generate predictions using Vertex AI Gemini,
    and save the results.
    """
    # --- Load Environment Variables ---
    load_dotenv()
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
    LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
    MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-1.0-pro") # Default model

    if not PROJECT_ID or not LOCATION:
        print("Error: GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION environment variables must be set.")
        return

    # --- Determine Prediction Field Name ---
    # Make it dynamic based on model and prompt key
    prompt_key_suffix = args.prompt_key.replace('prompt_', '') # e.g., '5shot' or '0shot'
    PREDICTION_FIELD_NAME = f"pred_{prompt_key_suffix}_{MODEL_NAME.lower().replace('-', '_')}"

    print("--- Configuration ---")
    print(f"Dataset Source: {args.dataset_source}")
    print(f"Is Local JSON: {args.is_local_json}")
    print(f"Split Name: {args.split_name if not args.is_local_json else 'N/A (local JSON)'}")
    print(f"Output Directory (Dataset format): {args.output_dir}")
    print(f"Output JSON for Eval (Optional): {args.save_eval_json_path}") # New config print
    print(f"Number of Samples: {args.num_samples if args.num_samples > 0 else 'All'}")
    print(f"Prompt Key: {args.prompt_key}")
    print(f"Prediction Field Name: {PREDICTION_FIELD_NAME}")
    print(f"Vertex Project: {PROJECT_ID}")
    print(f"Vertex Location: {LOCATION}")
    print(f"Vertex Model: {MODEL_NAME}")
    print("---------------------")

    # --- Load Dataset ---
    print("\n--- Loading Dataset ---")
    loaded_data = None
    try:
        if args.is_local_json:
            # Load from the custom JSON format
            loaded_data = create_dataset_from_custom_json(args.dataset_source)
            if loaded_data is None:
                 print("Failed to load dataset from local JSON.")
                 return # Exit if loading failed
            print(f"Successfully loaded custom JSON dataset. Processing {'all' if args.num_samples <= 0 else args.num_samples} samples.")
        else:
            # Load from Hugging Face Hub or a standard local dataset directory
            # Try loading a specific split first
            try:
                loaded_data = load_dataset(args.dataset_source, split=args.split_name)
                print(f"Successfully loaded split '{args.split_name}' from '{args.dataset_source}'. Processing {'all' if args.num_samples <= 0 else args.num_samples} samples.")
            except ValueError as e:
                 print(f"Warning: Could not load specific split '{args.split_name}'. Attempting to load the whole dataset. Error: {e}")
                 # If split loading fails, try loading the whole DatasetDict
                 dataset_dict = load_dataset(args.dataset_source)
                 if isinstance(dataset_dict, DatasetDict) and args.split_name in dataset_dict:
                     loaded_data = dataset_dict[args.split_name]
                     print(f"Successfully loaded split '{args.split_name}' after loading DatasetDict.")
                 elif isinstance(dataset_dict, Dataset):
                     loaded_data = dataset_dict # It might be a single Dataset object already
                     print("Loaded data as a single Dataset object (no specific split requested or found).")
                 else:
                     print(f"Error: Could not find split '{args.split_name}' in the loaded dataset dict keys: {list(dataset_dict.keys())}")
                     return

    except Exception as e:
        print(f"Error loading dataset '{args.dataset_source}': {e}")
        return

    if loaded_data is None:
        print("Dataset could not be loaded.")
        return

    # Ensure loaded_data is a Dataset object
    if not isinstance(loaded_data, Dataset):
        print(f"Error: Loaded data is not a Dataset object (Type: {type(loaded_data)}). Check source and split name.")
        return

    # --- Initialize Vertex AI ---
    print("\n--- Initializing Vertex AI ---")
    try:
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        model = GenerativeModel(MODEL_NAME)
        print(f"Vertex AI initialized. Using model: {MODEL_NAME}")
    except Exception as e:
        print(f"Error initializing Vertex AI: {e}")
        return

    # --- Process Samples ---
    print("\n--- Processing Samples ---")
    # This list will hold the dictionaries for our new dataset/JSON output
    processed_records_list = []
    num_to_process = len(loaded_data) if args.num_samples <= 0 else min(args.num_samples, len(loaded_data))

    for i in range(num_to_process):
        sample = loaded_data[i]
        # Determine sample ID - look for 'id' key, fallback to index
        sample_id = sample.get('id', f"index_{i}")

        print(f"\nProcessing Sample {i+1}/{num_to_process} (ID: {sample_id})")

        # Extract the prompt using the specified key
        if args.prompt_key not in sample or not sample[args.prompt_key]:
            print(f"Warning: Prompt key '{args.prompt_key}' not found or empty in sample {sample_id}. Skipping prediction.")
            generated_text = "ERROR: Prompt key missing or empty"
        else:
            prompt_text = sample[args.prompt_key]
            generated_text = None # Initialize prediction variable
            try:
                print("Sending prompt to Gemini...")
                # Configure generation parameters
                generation_config = {
                    "max_output_tokens": 2048, # Increased token limit slightly
                    "temperature": 0.2,      # Low temp for deterministic planning
                    "top_p": 0.8,
                    "top_k": 40
                }

                response = model.generate_content(
                    prompt_text,
                    generation_config=generation_config,
                    # safety_settings=... # Add safety settings if needed
                )
                # Check for valid response content
                if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                     generated_text = response.text.strip()
                     print(f"Prediction received successfully.")
                else:
                     # Handle cases where the response might be blocked or empty
                     finish_reason = response.candidates[0].finish_reason if response and response.candidates else "Unknown"
                     safety_ratings = response.candidates[0].safety_ratings if response and response.candidates else "N/A"
                     print(f"Warning: Received empty or blocked response for sample {sample_id}. Finish Reason: {finish_reason}, Safety Ratings: {safety_ratings}")
                     generated_text = f"ERROR: Received empty or blocked response (Reason: {finish_reason})"


                # Optional delay to avoid rate limits
                # time.sleep(0.5) # Reduced sleep time slightly

            except Exception as e:
                print(f"!!! Error generating prediction for sample {sample_id}: {e}")
                generated_text = f"ERROR: {e}" # Store error message

        # Create record for the new dataset/JSON
        new_record = sample.copy()
        new_record[PREDICTION_FIELD_NAME] = generated_text
        processed_records_list.append(new_record)

    print(f"\n--- Finished processing {len(processed_records_list)} samples ---")

    # --- Create and Save New Dataset (Arrow format) ---
    if processed_records_list:
        print("\nCreating new Dataset object (Arrow format)...")
        # Preserve original features if possible, otherwise infer
        original_features = loaded_data.features if hasattr(loaded_data, 'features') else None
        try:
            # Add the new prediction field to the features if defining explicitly
            if original_features:
                 new_features_dict = original_features.copy()
                 # Avoid overwriting if somehow the field already existed
                 if PREDICTION_FIELD_NAME not in new_features_dict:
                      new_features_dict[PREDICTION_FIELD_NAME] = Value('string')
                 new_dataset_features = Features(new_features_dict)
                 new_dataset = Dataset.from_list(processed_records_list, features=new_dataset_features)
                 print("New dataset created preserving and adding features.")
            else:
                 new_dataset = Dataset.from_list(processed_records_list)
                 print("New dataset created with inferred features.")

            print("\nNew Dataset Schema:")
            print(new_dataset.info)
            print(new_dataset.features) # Print detailed features

            # Ensure the output directory exists
            os.makedirs(args.output_dir, exist_ok=True)

            print(f"\nSaving new dataset (Arrow format) to disk at: {args.output_dir}")
            new_dataset.save_to_disk(args.output_dir)
            print("Dataset (Arrow format) saved successfully.")

        except Exception as e:
            print(f"Error creating or saving the new dataset (Arrow format): {e}")

        # --- Save JSON for Evaluation Script (Optional) ---
        if args.save_eval_json_path:
            print(f"\nSaving JSON output for evaluation script to: {args.save_eval_json_path}")
            # Convert the list of records back to the {id: record} format
            eval_json_data = {}
            for record in processed_records_list:
                record_id = record.get('id')
                if record_id is not None:
                    # Make a copy to avoid modifying the list's dicts if needed later
                    eval_json_data[record_id] = record.copy()
                else:
                    print("Warning: Record missing 'id' field, cannot include in evaluation JSON.")

            if eval_json_data:
                try:
                    # Ensure the directory for the JSON file exists
                    eval_json_dir = os.path.dirname(args.save_eval_json_path)
                    if eval_json_dir: # Check if path includes a directory
                         os.makedirs(eval_json_dir, exist_ok=True)

                    with open(args.save_eval_json_path, 'w', encoding='utf-8') as f:
                        json.dump(eval_json_data, f, indent=2) # Use indent for readability
                    print("Evaluation JSON saved successfully.")
                except Exception as e:
                    print(f"Error saving evaluation JSON to {args.save_eval_json_path}: {e}")
            else:
                print("No records with IDs found, skipping evaluation JSON saving.")

    else:
        print("\nNo records were processed, skipping dataset creation and saving.")

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate predictions using Gemini and save to a new dataset.")

    parser.add_argument("--dataset_source", type=str,
                        help="Path to a local JSON file or dataset directory, or name of a dataset on Hugging Face Hub.")
    parser.add_argument("--output_dir", type=str,
                        help="Directory path to save the processed dataset (Arrow format).")

    parser.add_argument("--is_local_json", action="store_true",
                        help="Flag indicating that 'dataset_source' is a path to a local custom JSON file (dict of dicts format).")
    parser.add_argument("--split_name", type=str, default="train",
                        help="Name of the dataset split to process (e.g., 'train', 'test', 'validation'). Ignored if --is_local_json is set.")
    parser.add_argument("--num_samples", type=int, default=5,
                        help="Number of samples to process. Set to 0 or negative to process all samples. Default is 5.")
    parser.add_argument("--prompt_key", type=str, default="prompt_5shot",
                        help="The key in the dataset sample dictionary containing the text prompt for the model. Default is 'prompt_5shot'.")
    parser.add_argument("--save_eval_json_path", type=str, default=None, # New argument
                        help="Optional path to save the results as a single JSON file compatible with the evaluation script.")

    # Example: python process_data.py clembench-playpen/natural-plan-meeting ./output_data --split_name train --num_samples 10 --save_eval_json_path ./output_data/eval_ready.json
    # Example: python process_data.py ./my_custom_data.json ./output_data --is_local_json --num_samples 50 --prompt_key my_prompt_field --save_eval_json_path ./output_data/eval_ready.json

    parsed_args = parser.parse_args()
    main(parsed_args)
