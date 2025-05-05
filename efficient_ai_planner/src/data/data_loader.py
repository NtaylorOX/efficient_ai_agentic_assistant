import json
import os
from datasets import Dataset, Features, Value 

def create_dataset_from_custom_json(data_path: str, save_path: str, features: Features = None) -> Dataset | None:
    """
    Loads data from a JSON file with a specific structure (dict of dicts),
    transforms it into a Hugging Face Dataset, and saves it to disk.

    The expected JSON structure is:
    {
        "example_id_1": {"field1": value1, "field2": value2, ...},
        "example_id_2": {"field1": valueA, "field2": valueB, ...},
        ...
    }

    Args:
        data_path (str): The path to the input JSON file.
        save_path (str): The directory path where the Hugging Face Dataset
                         will be saved. This directory will be created if
                         it doesn't exist.
        features (datasets.Features, optional): An optional Features object
            to define the dataset schema explicitly. If None, features will
            be inferred. Defaults to None.

    Returns:
        datasets.Dataset | None: The created Dataset object if successful,
                                 otherwise None if loading fails.
    """
    # --- 1. Load JSON Data ---
    print(f"Attempting to load JSON data from: {data_path}")
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
        print(f"An unexpected error occurred during loading: {e}")
        return None

    # --- 2. Transform Data ---
    print("Transforming data into a list of records...")
    data_list = []
    for example_id, record_data in data_dict.items():
        if isinstance(record_data, dict):
            new_record = record_data.copy()
            new_record['id'] = example_id  # Add the original key as 'id'
            data_list.append(new_record)
        else:
            print(f"Warning: Skipping key '{example_id}' as its value is not a dictionary.")

    if not data_list:
        print("Error: No valid records found after transformation.")
        return None

    print(f"Transformed {len(data_list)} records.")

    # --- 3. Create Dataset Object ---
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
    except Exception as e:
        print(f"Error creating Dataset object: {e}")
        return None

    # --- 4. Save Dataset to Disk ---
    print(f"\nAttempting to save dataset to disk at: {save_path}")
    try:
        if save_path:
            # Ensure the directory exists, create if not
            os.makedirs(save_path, exist_ok=True)
            custom_dataset.save_to_disk(save_path)
            print(f"Dataset successfully saved to {save_path}")
    except Exception as e:
        print(f"Error saving dataset to disk: {e}")
        # We still return the dataset object even if saving fails,
        # as it exists in memory. The user might want to handle saving differently.

    # --- 5. Return Dataset ---
    return custom_dataset

