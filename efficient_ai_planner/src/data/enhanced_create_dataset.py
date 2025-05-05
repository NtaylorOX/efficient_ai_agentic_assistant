# --- Imports ---
import collections
import datetime
import json
import ast  # For safely evaluating string representations
import os
from typing import Any, Sequence, List, Dict, Optional, Tuple

# Using absl-py for flags and app execution
# Install: pip install absl-py
from absl import app
from absl import flags
from absl import logging

# --- Flag Definitions ---
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_data_path",
    None,
    "Path to the input JSON data file (dict of dicts format, like the one used by the evaluator).",
    required=True,
)
flags.DEFINE_string(
    "output_jsonl_path",
    None,
    "Path to save the generated synthetic dataset (JSON Lines format).",
    required=True,
)
flags.DEFINE_integer(
    "max_examples",
    0, # 0 means process all examples
    "Maximum number of examples from the input file to process. 0 for all.",
)


# --- Helper Functions (Adapted from Evaluator) ---

def safe_literal_eval(value: Any) -> Any:
    """Safely evaluate a string representation of a Python literal."""
    if isinstance(value, str):
        try:
            # Basic check to prevent evaluating overly complex/long strings
            if len(value) > 10000:
                 logging.warning("Skipping evaluation of very long string.")
                 return value
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError, MemoryError):
            # Log warning if evaluation fails
            logging.warning(f"Could not evaluate string literal: {value[:100]}...")
            return value # Return original string
    return value # Return original value if not a string

def convert_to_time_obj(time_str: str) -> Optional[datetime.datetime]:
    """Converts HH:MM AM/PM string to datetime object. Handles potential errors."""
    if not isinstance(time_str, str):
        logging.error(f"Invalid time input (not a string): {time_str}")
        return None
    try:
        # Handle potential extra spaces
        return datetime.datetime.strptime(time_str.strip(), "%I:%M%p")
    except ValueError:
        logging.error(f"Invalid time format: '{time_str}'. Expected HH:MM AM/PM.")
        return None

def format_time_obj(dt_obj: Optional[datetime.datetime]) -> str:
    """Formats a datetime object back to HH:MM AM/PM string."""
    if dt_obj is None:
        return "Invalid Time"
    return dt_obj.strftime("%I:%M%p").lstrip('0') # Remove leading zero from hour if present

def format_constraints_for_prompt(constraints_info: Dict[str, Any]) -> str:
    """Formats the processed constraints into a readable string for the prompt."""
    if not constraints_info or 'person_constraints' not in constraints_info:
        return "No constraints available."

    lines = []
    start_location = constraints_info.get('start_location', 'Unknown')
    initial_time_str = format_time_obj(constraints_info.get('initial_time'))
    lines.append(f"Start: {start_location} at {initial_time_str}")
    lines.append("Constraints:")
    for name, details in constraints_info.get('person_constraints', {}).items():
        location = details.get('location', 'Unknown')
        start_time_str = format_time_obj(details.get('start_time'))
        end_time_str = format_time_obj(details.get('end_time'))
        duration = details.get('meeting_time', 'N/A')
        lines.append(f"- {name}: At {location} from {start_time_str} to {end_time_str}, for {duration} minutes.")
    return "\n".join(lines)

def format_dist_matrix_for_prompt(dist_matrix: Dict[str, Dict[str, int]]) -> str:
    """Formats the distance matrix into a readable string for the prompt."""
    if not dist_matrix:
        return "No distance information available."
    lines = ["Distances (minutes):"]
    for origin, destinations in dist_matrix.items():
        for destination, time in destinations.items():
            lines.append(f"- {origin} to {destination}: {time}")
    return "\n".join(lines)

def process_constraints(raw_constraints_data: Any) -> Optional[Dict[str, Any]]:
    """Processes raw constraint data into a structured dictionary."""
    # Note: This function now directly returns the structured dict or None
    constraints_list = safe_literal_eval(raw_constraints_data)
    if not isinstance(constraints_list, list):
        logging.error(f"Constraints data is not a list: {type(constraints_list)}")
        return None
    if len(constraints_list) < 1: # Need at least start info
        logging.error("Constraints list is empty.")
        return None

    # Extract start location and time
    start_info = constraints_list[0]
    if not isinstance(start_info, (list, tuple)) or len(start_info) != 2:
        logging.error(f"Invalid start info format: {start_info}")
        return None
    start_location, initial_time_str = start_info
    initial_time = convert_to_time_obj(initial_time_str)
    if initial_time is None: return None # Error logged in convert_to_time_obj

    # Process individual constraints (people to meet)
    person_constraints_processed = {}
    if len(constraints_list) > 1:
        for item in constraints_list[1:]:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                logging.warning(f"Skipping invalid constraint format: {item}")
                continue

            name, location, times_str, meeting_time_minutes = item

            # Validate and parse time range string
            if not isinstance(times_str, str) or "to" not in times_str:
                 logging.warning(f"Skipping constraint for '{name}' due to invalid time range string: {times_str}")
                 continue
            try:
                start_time_str, end_time_str = times_str.split("to")
                start_time = convert_to_time_obj(start_time_str)
                end_time = convert_to_time_obj(end_time_str)
                if start_time is None or end_time is None: raise ValueError("Invalid time conversion")
            except ValueError as e:
                logging.warning(f"Skipping constraint for '{name}' due to time parsing error: {e} in '{times_str}'")
                continue

            # Validate meeting time
            try:
                meeting_time = int(meeting_time_minutes)
            except (ValueError, TypeError):
                logging.warning(f"Skipping constraint for '{name}' due to invalid meeting time: {meeting_time_minutes}")
                continue

            person_constraints_processed[name] = {
                "location": location,
                "start_time": start_time,
                "end_time": end_time,
                "meeting_time": meeting_time,
            }

    # Return structured constraints info
    return {
        "start_location": start_location,
        "initial_time": initial_time,
        "person_constraints": person_constraints_processed
    }

def parse_golden_plan(raw_golden_plan: Any) -> Optional[List[str]]:
    """Parses the golden plan, handling string list representation."""
    golden_plan_parsed = safe_literal_eval(raw_golden_plan)

    if isinstance(golden_plan_parsed, list):
        # Ensure all elements are strings
        return [str(step) for step in golden_plan_parsed if str(step).strip()]
    elif isinstance(golden_plan_parsed, str):
        # If it's a single string, try splitting by common delimiters like newline or '.'
        # This is less ideal than a proper list format in the source data
        steps = [step.strip() for step in golden_plan_parsed.replace('\n', '.').split('.') if step.strip()]
        if steps:
            logging.warning("Parsed golden plan from a single string, format might be inconsistent.")
            return steps
        else:
             logging.error("Could not parse golden plan from string.")
             return None
    else:
        logging.error(f"Invalid golden plan format: {type(golden_plan_parsed)}")
        return None


# --- Synthetic Data Generation Logic ---

def generate_synthetic_steps(
    example_id: str,
    constraints_info: Dict[str, Any],
    dist_matrix: Dict[str, Dict[str, int]],
    golden_plan_steps: List[str]
) -> List[Dict[str, str]]:
    """
    Generates step-by-step training instances from a single example.
    Returns a list of {"prompt": ..., "completion": ...} dictionaries.
    """
    synthetic_data = []
    if not golden_plan_steps:
        logging.warning(f"[{example_id}] No golden plan steps to process.")
        return []

    # Initial state from constraints
    cur_location = constraints_info['start_location']
    cur_time = constraints_info['initial_time']
    plan_history = [] # Keep track of the steps taken *in the golden plan format*

    # Format the static parts of the prompt
    problem_desc = format_constraints_for_prompt(constraints_info)
    distances_desc = format_dist_matrix_for_prompt(dist_matrix)

    # Simulate the golden plan
    for i, step_text in enumerate(golden_plan_steps):
        # --- Formulate the prompt for the *current* state ---
        current_state_desc = f"Current State:\nLocation: {cur_location}\nTime: {format_time_obj(cur_time)}"
        history_desc = "Plan History:\n" + ("\n".join(f"- {h}" for h in plan_history) if plan_history else " (None)")

        prompt = (
            f"{problem_desc}\n\n"
            f"{distances_desc}\n\n"
            f"{current_state_desc}\n\n"
            f"{history_desc}\n\n"
            "What is the next single step in the plan?"
        )

        # The completion is the current step from the golden plan
        completion = step_text

        # Add to synthetic dataset
        synthetic_data.append({"prompt": prompt, "completion": completion})

        # --- Update state based on the current step_text ---
        # This logic mimics the validator to advance time/location accurately
        try:
            if step_text.startswith("You start"):
                # State is already initialized, just add to history
                pass
            elif step_text.startswith("You travel"):
                parts = step_text.split(" travel to ")
                if len(parts) < 2: raise ValueError("Malformed travel step")
                dest_part = parts[1].split(" in")[0].strip()
                destination = dest_part

                if cur_location not in dist_matrix or destination not in dist_matrix.get(cur_location, {}):
                    raise ValueError(f"Missing distance from '{cur_location}' to '{destination}'")

                travel_time = dist_matrix[cur_location][destination]
                cur_time += datetime.timedelta(minutes=travel_time)
                cur_location = destination
            elif step_text.startswith("You wait"):
                parts = step_text.split("wait until ")
                if len(parts) < 2: raise ValueError("Malformed wait step")
                raw_end_time = parts[1].split(".")[0].strip()
                end_time = convert_to_time_obj(raw_end_time)
                if end_time is None: raise ValueError("Invalid wait time format")

                if end_time < cur_time:
                     logging.warning(f"[{example_id}] Wait step time {format_time_obj(end_time)} is before current time {format_time_obj(cur_time)}. Clamping to current time.")
                     cur_time = cur_time # Effectively zero wait if target is in past
                else:
                     cur_time = end_time
            elif step_text.startswith("You meet"):
                 # We need the duration from the *constraints* to update time correctly
                 parts = step_text.split("meet ")
                 if len(parts) < 2: raise ValueError("Malformed meet step")
                 person = parts[1].split(" for")[0].strip()

                 if person not in constraints_info.get('person_constraints', {}):
                      raise ValueError(f"Person '{person}' mentioned in plan but not in constraints.")

                 meeting_duration = constraints_info['person_constraints'][person]['meeting_time']
                 cur_time += datetime.timedelta(minutes=meeting_duration)
                 # Location doesn't change during the meeting itself based on this plan format
            else:
                # Handle unrecognized step format
                raise ValueError(f"Unknown plan step format: '{step_text[:50]}...'")

            # Add the processed step to history *after* generating the prompt for it
            plan_history.append(step_text)

        except ValueError as e:
            logging.error(f"[{example_id}] Error processing step '{step_text}': {e}. Stopping generation for this example.")
            # Stop generating further steps for this example if an error occurs
            break
        except KeyError as e:
             logging.error(f"[{example_id}] Missing key during step processing '{step_text}': {e}. Stopping generation.")
             break


    # Optionally, add a final "end of plan" step?
    # Could be useful for teaching the model when to stop.
    # if plan_history: # Only add if some steps were processed
    #     current_state_desc = f"Current State:\nLocation: {cur_location}\nTime: {format_time_obj(cur_time)}"
    #     history_desc = "Plan History:\n" + ("\n".join(f"- {h}" for h in plan_history) if plan_history else " (None)")
    #     prompt = (
    #         f"{problem_desc}\n\n"
    #         f"{distances_desc}\n\n"
    #         f"{current_state_desc}\n\n"
    #         f"{history_desc}\n\n"
    #         "What is the next single step in the plan?"
    #     )
    #     completion = "END_PLAN" # Or just "" or some specific token
    #     synthetic_data.append({"prompt": prompt, "completion": completion})


    return synthetic_data


# --- Main Execution Logic ---

def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Load the input JSON data
    logging.info(f"Loading input data from: {FLAGS.input_data_path}")
    try:
        with open(FLAGS.input_data_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        logging.info(f"Loaded {len(input_data)} records from input file.")
    except FileNotFoundError:
        logging.error(f"Input data file not found: {FLAGS.input_data_path}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding input JSON: {e}")
        return

    # Prepare output file
    logging.info(f"Will save synthetic data to: {FLAGS.output_jsonl_path}")
    output_dir = os.path.dirname(FLAGS.output_jsonl_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    total_synthetic_steps = 0
    examples_processed = 0

    try:
        with open(FLAGS.output_jsonl_path, 'w', encoding='utf-8') as outfile:
            # Iterate through data items
            for example_id, obj in input_data.items():
                if FLAGS.max_examples > 0 and examples_processed >= FLAGS.max_examples:
                    logging.info(f"Reached max examples limit ({FLAGS.max_examples}). Stopping.")
                    break
                examples_processed += 1

                logging.info(f"--- Processing Example: {example_id} ({examples_processed}/{len(input_data) if FLAGS.max_examples <= 0 else FLAGS.max_examples}) ---")

                # --- Get and Parse Data ---
                raw_constraints = obj.get("constraints")
                raw_dist_matrix = obj.get("dist_matrix")
                raw_golden_plan = obj.get("golden_plan")

                if not all([raw_constraints, raw_dist_matrix, raw_golden_plan]):
                    logging.warning(f"Skipping {example_id}: Missing one or more required fields (constraints, dist_matrix, golden_plan).")
                    continue

                constraints_info = process_constraints(raw_constraints)
                if constraints_info is None:
                    logging.warning(f"Skipping {example_id}: Failed to process constraints.")
                    continue

                dist_matrix = safe_literal_eval(raw_dist_matrix)
                if not isinstance(dist_matrix, dict):
                     logging.warning(f"Skipping {example_id}: Invalid distance matrix format.")
                     continue

                golden_plan_steps = parse_golden_plan(raw_golden_plan)
                if golden_plan_steps is None:
                    logging.warning(f"Skipping {example_id}: Failed to parse golden plan.")
                    continue

                # --- Generate Synthetic Steps ---
                synthetic_steps = generate_synthetic_steps(
                    example_id, constraints_info, dist_matrix, golden_plan_steps
                )

                # --- Write to Output File ---
                if synthetic_steps:
                    logging.info(f"Generated {len(synthetic_steps)} synthetic steps for {example_id}.")
                    for step_data in synthetic_steps:
                        try:
                            outfile.write(json.dumps(step_data) + '\n')
                            total_synthetic_steps += 1
                        except TypeError as e:
                             logging.error(f"[{example_id}] Error serializing step data to JSON: {e} | Data: {step_data}")
                else:
                    logging.warning(f"No synthetic steps generated for {example_id}.")

    except IOError as e:
        logging.error(f"Error writing to output file {FLAGS.output_jsonl_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred during processing: {e}", exc_info=True) # Log traceback

    print("\n--- Synthetic Data Generation Summary ---")
    print(f"Processed {examples_processed} examples from input.")
    print(f"Generated a total of {total_synthetic_steps} synthetic steps.")
    print(f"Output saved to: {FLAGS.output_jsonl_path}")


if __name__ == "__main__":
  app.run(main)

