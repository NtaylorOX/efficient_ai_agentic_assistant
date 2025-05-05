# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Eval Script for Meeting Planning."""

import collections
import datetime
import json
import ast  # For safely evaluating string representations
from typing import Any, Sequence, List, Dict, Union

# Using absl-py for flags and app execution
# Install: pip install absl-py
from absl import app
from absl import flags
from absl import logging # Use absl logging

# --- Flag Definitions ---
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "data_path",
    None, # No default, must be provided
    "Path to the input JSON data file containing model responses.",
    required=True,
)
flags.DEFINE_string(
    "prediction_field_name",
    "pred_5shot_gemini_1_0_pro", # Example default, adjust as needed or provide via command line
    "The exact field name in the JSON file that contains the model's predicted plan text.",
)
flags.DEFINE_integer(
    "max_constraints_eval",
    10,
    "Maximum number of constraints (people) to evaluate accuracy for.",
)
flags.DEFINE_integer(
    "num_samples_per_constraint_group", # Renamed for clarity
    100, # Assuming 100 samples per group as in original script
    "Expected number of samples per 'num_people' group for calculating grouped accuracy.",
)


# --- Helper Functions ---

def safe_literal_eval(value: Any) -> Any:
    """Safely evaluate a string representation of a Python literal."""
    if isinstance(value, str):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError):
            logging.warning(f"Could not evaluate string literal: {value[:100]}...") # Log warning
            return value # Return original string if evaluation fails
    return value # Return original value if not a string

def convert_to_time_obj(time_str: str) -> datetime.datetime | None:
    """Converts HH:MM AM/PM string to datetime object. Handles potential errors."""
    if not isinstance(time_str, str):
        logging.error(f"Invalid time input (not a string): {time_str}")
        return None
    try:
        return datetime.datetime.strptime(time_str.strip(), "%I:%M%p")
    except ValueError:
        logging.error(f"Invalid time format: '{time_str}'. Expected HH:MM AM/PM.")
        return None

def process_constraints(raw_constraints_data: Any) -> Dict[str, Dict[str, Any]] | None:
    """Processes raw constraint data into a structured dictionary."""
    constraints_list = safe_literal_eval(raw_constraints_data)
    if not isinstance(constraints_list, list) or len(constraints_list) < 2:
         logging.error(f"Constraints data is not a valid list or too short: {raw_constraints_data}")
         return None # Invalid format

    # Extract start location and time (handle potential errors)
    start_info = constraints_list[0]
    if not isinstance(start_info, (list, tuple)) or len(start_info) != 2:
        logging.error(f"Invalid start info format: {start_info}")
        return None
    start_location, initial_time_str = start_info
    initial_time = convert_to_time_obj(initial_time_str)
    if initial_time is None: return None # Error handled in convert_to_time_obj

    # Process individual constraints
    processed = collections.defaultdict(dict)
    for item in constraints_list[1:]:
        if not isinstance(item, (list, tuple)) or len(item) != 4:
            logging.warning(f"Skipping invalid constraint format: {item}")
            continue # Skip malformed constraint

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

        # Validate meeting time (should be number)
        try:
            meeting_time = int(meeting_time_minutes)
        except (ValueError, TypeError):
            logging.warning(f"Skipping constraint for '{name}' due to invalid meeting time: {meeting_time_minutes}")
            continue

        processed[name]["location"] = location
        processed[name]["start_time"] = start_time
        processed[name]["end_time"] = end_time
        processed[name]["meeting_time"] = meeting_time

    if not processed:
        logging.error("No valid individual constraints processed.")
        return None

    # Return structured constraints and start info separately
    return {
        "start_location": start_location,
        "initial_time": initial_time,
        "person_constraints": dict(processed) # Convert defaultdict back to dict
    }


def parse_text_plan(plan_text: str) -> List[str]:
    """
    Parses the model's text plan into a list of steps.
    Handles 'SOLUTION:' prefix and potential error messages.
    """
    if not isinstance(plan_text, str):
        logging.warning("Received non-string plan text, returning empty plan.")
        return []

    # Handle explicit error messages from generation script
    if plan_text.strip().startswith("ERROR:"):
        logging.warning(f"Plan text indicates an error: {plan_text}")
        return []

    prefix = "SOLUTION:"
    if prefix in plan_text:
        plan_text = plan_text[plan_text.find(prefix) + len(prefix) :]

    # Split by '.', strip whitespace, and filter out empty strings
    steps = [step.strip() for step in plan_text.split('.') if step.strip()]
    return steps


def validator_from_text(
    plan_steps: List[str],
    constraints_info: Dict[str, Any],
    dist_matrix: Dict[str, Dict[str, int]],
) -> int:
    """
    Computes the number of valid meetings scheduled in the plan (parsed text format).
    Returns the count of successfully scheduled meetings according to constraints.
    """
    if not plan_steps: # Handle empty plan from parser
        return 0
    if not constraints_info or 'person_constraints' not in constraints_info:
        logging.error("Validator received invalid constraints info.")
        return 0 # Cannot validate without constraints

    person_constraints = constraints_info['person_constraints']
    start_location = constraints_info['start_location']
    cur_time = constraints_info['initial_time']
    cur_location = start_location

    met_with = set() # Use a set for efficient checking
    score = 0

    for step in plan_steps:
        try:
            if step.startswith("You start"):
                # Optional: Validate start location/time if needed, but usually assumed correct
                continue
            elif step.startswith("You travel"):
                parts = step.split(" travel to ")
                if len(parts) < 2: raise ValueError("Malformed travel step")
                dest_part = parts[1].split(" in")[0].strip()
                destination = dest_part

                if cur_location not in dist_matrix or destination not in dist_matrix[cur_location]:
                    raise ValueError(f"Missing distance from '{cur_location}' to '{destination}'")

                travel_time = dist_matrix[cur_location][destination]
                cur_time += datetime.timedelta(minutes=travel_time)
                cur_location = destination
            elif step.startswith("You wait"):
                parts = step.split("wait until ")
                if len(parts) < 2: raise ValueError("Malformed wait step")
                raw_end_time = parts[1].split(".")[0].strip() # Remove potential trailing dot
                end_time = convert_to_time_obj(raw_end_time)
                if end_time is None: raise ValueError("Invalid wait time format")

                if end_time < cur_time:
                    # Allow waiting if end time is *equal* to current time (0 min wait)
                    logging.warning(f"Wait step goes backwards or has zero duration: cur={cur_time}, wait_until={end_time}. Allowing.")
                    # raise ValueError("Cannot go backwards in time") # Make it a warning or allow zero wait
                cur_time = end_time # Update time even if it's a zero wait
            elif step.startswith("You meet"):
                parts = step.split("meet ")
                if len(parts) < 2: raise ValueError("Malformed meet step")
                person_part = parts[1].split(" for")[0].strip()
                person = person_part

                if person in met_with:
                    raise ValueError(f"Person '{person}' already met with.")
                if person not in person_constraints:
                     raise ValueError(f"Person '{person}' not found in constraints.")

                # Check constraints
                constraint = person_constraints[person]
                required_meeting_time = constraint["meeting_time"]
                expected_location = constraint["location"]
                available_start = constraint["start_time"]
                available_end = constraint["end_time"]

                # Calculate meeting end time based on *required* duration
                meeting_end_time = cur_time + datetime.timedelta(minutes=required_meeting_time)

                # Validate location, availability window, and meeting end time
                if (
                    cur_location == expected_location
                    and cur_time >= available_start
                    and meeting_end_time <= available_end # Meeting must END within window
                ):
                    score += 1
                    cur_time = meeting_end_time # Advance time by meeting duration
                    met_with.add(person) # Mark person as met
                else:
                    # Provide more specific error reason
                    reason = []
                    if cur_location != expected_location: reason.append(f"wrong location (expected {expected_location}, was {cur_location})")
                    if cur_time < available_start: reason.append(f"too early (available from {available_start})")
                    if meeting_end_time > available_end: reason.append(f"too late (must finish by {available_end})")
                    raise ValueError(f"Invalid meeting for '{person}': {'; '.join(reason)}")
            else:
                # Handle unrecognized step format
                raise ValueError(f"Unknown plan step format: '{step[:50]}...'")

        except ValueError as e:
            logging.warning(f"Validation error: {e} | Step: '{step}' | Stopping validation for this plan.")
            # Stop processing this plan on first error
            return score # Return score accumulated so far

    # Return final score if all steps processed without critical error
    return score


# --- Main Execution Logic ---

def main(argv: Sequence[str]) -> None:
    # Basic argument check (absl handles required flags)
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Load the JSON data
    logging.info(f"Loading data from: {FLAGS.data_path}")
    try:
        with open(FLAGS.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Loaded {len(data)} records.")
    except FileNotFoundError:
        logging.error(f"Data file not found: {FLAGS.data_path}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return

    # Initialize accumulators
    # Use defaultdict for easier accumulation
    correct_per_num_people = collections.defaultdict(int)
    total_per_num_people = collections.defaultdict(int)
    total_correct = 0
    total_processed = 0

    # Iterate through data items (assuming dict format {id: record})
    for example_id, obj in data.items():
        total_processed += 1
        logging.info(f"Processing example: {example_id}")

        # Safely get required fields
        raw_constraints = obj.get("constraints")
        raw_dist_matrix = obj.get("dist_matrix")
        raw_golden_plan = obj.get("golden_plan") # Might be string or list
        pred_plan_text = obj.get(FLAGS.prediction_field_name) # Use flag value
        num_people = obj.get("num_people")

        # --- Data Validation and Parsing ---
        # Validate num_people
        try:
            num_people = int(num_people)
            if not 1 <= num_people <= FLAGS.max_constraints_eval:
                 logging.warning(f"Skipping {example_id}: num_people ({num_people}) out of range [1, {FLAGS.max_constraints_eval}].")
                 continue
        except (ValueError, TypeError):
            logging.warning(f"Skipping {example_id}: Invalid 'num_people' value: {num_people}")
            continue

        # Process constraints
        constraints_info = process_constraints(raw_constraints)
        if constraints_info is None:
            logging.warning(f"Skipping {example_id}: Failed to process constraints.")
            continue

        # Process distance matrix
        dist_matrix = safe_literal_eval(raw_dist_matrix)
        if not isinstance(dist_matrix, dict):
             logging.warning(f"Skipping {example_id}: Invalid distance matrix format.")
             continue

        # Parse golden plan (handle string list representation)
        golden_plan_parsed = safe_literal_eval(raw_golden_plan)
        if isinstance(golden_plan_parsed, str): # If eval failed, try parsing as text
             golden_plan_steps = parse_text_plan(golden_plan_parsed)
        elif isinstance(golden_plan_parsed, list):
             golden_plan_steps = [str(step) for step in golden_plan_parsed] # Ensure steps are strings
        else:
             logging.warning(f"Skipping {example_id}: Invalid golden plan format: {type(golden_plan_parsed)}")
             continue

        # Parse predicted plan
        pred_plan_steps = parse_text_plan(pred_plan_text)

        # --- Validation ---
        logging.debug(f"Validating golden plan for {example_id}...")
        golden_score = validator_from_text(
            golden_plan_steps, constraints_info, dist_matrix
        )
        logging.debug(f"Golden score: {golden_score}")

        logging.debug(f"Validating predicted plan for {example_id}...")
        pred_score = validator_from_text(
            pred_plan_steps, constraints_info, dist_matrix
        )
        logging.debug(f"Predicted score: {pred_score}")

        # --- Accuracy Calculation ---
        # Consider a prediction correct if it achieves the same score as the golden plan
        is_correct = (pred_score == golden_score)

        # Accumulate results
        total_per_num_people[num_people] += 1
        if is_correct:
            correct_per_num_people[num_people] += 1
            total_correct += 1

        logging.info(f"Result for {example_id}: {'Correct' if is_correct else 'Incorrect'} (Pred: {pred_score}, Golden: {golden_score})")


    # --- Report Results ---
    print("\n--- Evaluation Results ---")
    print(f"Prediction field evaluated: '{FLAGS.prediction_field_name}'")
    print(f"Total examples processed: {total_processed}")
    print(f"Total overall correct: {total_correct}")

    # Calculate and print accuracy per group
    for n in range(1, FLAGS.max_constraints_eval + 1):
        total_in_group = total_per_num_people[n]
        correct_in_group = correct_per_num_people[n]
        # Use expected number if actual count is different? Original script divides by constant.
        # Let's report based on actual processed count for robustness.
        # expected_samples = FLAGS.num_samples_per_constraint_group
        if total_in_group > 0:
             accuracy = correct_in_group / total_in_group
             print(f"Accuracy for {n} people ({total_in_group} samples): {accuracy:.4f} ({correct_in_group}/{total_in_group})")
        else:
             print(f"Accuracy for {n} people (0 samples): N/A")

    # Calculate and print overall accuracy
    if total_processed > 0:
        overall_accuracy = total_correct / total_processed
        print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({total_correct}/{total_processed})")
    else:
        print("\nOverall Accuracy: N/A (No samples processed)")


if __name__ == "__main__":
  # Required flag check happens automatically via app.run
  app.run(main)

