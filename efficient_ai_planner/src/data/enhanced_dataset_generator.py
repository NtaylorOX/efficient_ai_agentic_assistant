# --- Imports ---
import collections
import datetime
import json
import ast  # For safely evaluating string representations
import os
from typing import Any, Sequence, List, Dict, Optional, Tuple

# Using absl-py for flags and app execution
# Install: pip install absl-py google-cloud-aiplatform python-dotenv
from absl import app
from absl import flags
from absl import logging
from dotenv import load_dotenv

# Vertex AI specific imports (only needed if using Gemini)
try:
    import vertexai
    from vertexai.generative_models import GenerativeModel, HarmCategory, HarmBlockThreshold
    VERTEX_AI_AVAILABLE = True
except ImportError:
    logging.warning("Vertex AI SDK not found. Install 'google-cloud-aiplatform' to use the Gemini CoT feature.")
    VERTEX_AI_AVAILABLE = False


# --- Flag Definitions ---
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_data_path",
    None,
    "Path to the input JSON data file (dict of dicts format).",
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
flags.DEFINE_boolean(
    "use_gemini_cot",
    False,
    "If True, use Gemini via Vertex AI to generate CoT reasoning/plans instead of decomposing the golden plan.",
)
flags.DEFINE_string(
    "prompt_key_for_gemini",
    "prompt_0shot", # Default to the zero-shot prompt for CoT generation
    "The key in the input data containing the prompt to feed to Gemini when use_gemini_cot is True.",
)
flags.DEFINE_string(
    "gemini_model_name",
    "gemini-2.0-flash", # Default model
    "The name of the Gemini model to use via Vertex AI (e.g., 'gemini-1.5-pro-preview-0409').",
)


# --- Helper Functions (Adapted/Added) ---

def safe_literal_eval(value: Any) -> Any:
    """Safely evaluate a string representation of a Python literal."""
    if isinstance(value, str):
        try:
            if len(value) > 10000:
                 logging.warning("Skipping evaluation of very long string.")
                 return value
            return ast.literal_eval(value)
        except (ValueError, SyntaxError, TypeError, MemoryError):
            logging.warning(f"Could not evaluate string literal: {value[:100]}...")
            return value
    return value

def convert_to_time_obj(time_str: str) -> Optional[datetime.datetime]:
    """Converts HH:MM AM/PM string to datetime object."""
    if not isinstance(time_str, str):
        logging.error(f"Invalid time input (not a string): {time_str}")
        return None
    try:
        return datetime.datetime.strptime(time_str.strip(), "%I:%M%p")
    except ValueError:
        logging.error(f"Invalid time format: '{time_str}'. Expected HH:MM AM/PM.")
        return None

def format_time_obj(dt_obj: Optional[datetime.datetime]) -> str:
    """Formats a datetime object back to HH:MM AM/PM string."""
    if dt_obj is None:
        return "Invalid Time"
    return dt_obj.strftime("%I:%M%p").lstrip('0')

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
    constraints_list = safe_literal_eval(raw_constraints_data)
    if not isinstance(constraints_list, list):
        logging.error(f"Constraints data is not a list: {type(constraints_list)}")
        return None
    if len(constraints_list) < 1:
        logging.error("Constraints list is empty.")
        return None
    start_info = constraints_list[0]
    if not isinstance(start_info, (list, tuple)) or len(start_info) != 2:
        logging.error(f"Invalid start info format: {start_info}")
        return None
    start_location, initial_time_str = start_info
    initial_time = convert_to_time_obj(initial_time_str)
    if initial_time is None: return None
    person_constraints_processed = {}
    if len(constraints_list) > 1:
        for item in constraints_list[1:]:
            if not isinstance(item, (list, tuple)) or len(item) != 4:
                logging.warning(f"Skipping invalid constraint format: {item}")
                continue
            name, location, times_str, meeting_time_minutes = item
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
            try:
                meeting_time = int(meeting_time_minutes)
            except (ValueError, TypeError):
                logging.warning(f"Skipping constraint for '{name}' due to invalid meeting time: {meeting_time_minutes}")
                continue
            person_constraints_processed[name] = {
                "location": location, "start_time": start_time,
                "end_time": end_time, "meeting_time": meeting_time,
            }
    return {
        "start_location": start_location, "initial_time": initial_time,
        "person_constraints": person_constraints_processed
    }

def parse_golden_plan(raw_golden_plan: Any) -> Optional[List[str]]:
    """Parses the golden plan, handling string list representation."""
    golden_plan_parsed = safe_literal_eval(raw_golden_plan)
    if isinstance(golden_plan_parsed, list):
        return [str(step) for step in golden_plan_parsed if str(step).strip()]
    elif isinstance(golden_plan_parsed, str):
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

def generate_synthetic_steps_from_golden(
    example_id: str,
    constraints_info: Dict[str, Any],
    dist_matrix: Dict[str, Dict[str, int]],
    golden_plan_steps: List[str]
) -> List[Dict[str, str]]:
    """
    Generates step-by-step training instances based on the golden plan.
    """
    synthetic_data = []
    if not golden_plan_steps:
        logging.warning(f"[{example_id}] No golden plan steps to process.")
        return []
    cur_location = constraints_info['start_location']
    cur_time = constraints_info['initial_time']
    plan_history = []
    problem_desc = format_constraints_for_prompt(constraints_info)
    distances_desc = format_dist_matrix_for_prompt(dist_matrix)

    for i, step_text in enumerate(golden_plan_steps):
        current_state_desc = f"Current State:\nLocation: {cur_location}\nTime: {format_time_obj(cur_time)}"
        history_desc = "Plan History:\n" + ("\n".join(f"- {h}" for h in plan_history) if plan_history else " (None)")
        prompt = (
            f"{problem_desc}\n\n"
            f"{distances_desc}\n\n"
            f"{current_state_desc}\n\n"
            f"{history_desc}\n\n"
            "What is the next single step in the plan?"
        )
        completion = step_text
        synthetic_data.append({"prompt": prompt, "completion": completion})

        try:
            # Update state logic (same as before)
            if step_text.startswith("You start"): pass
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
                cur_time = max(cur_time, end_time) # Clamp to current time if wait target is in past
            elif step_text.startswith("You meet"):
                 parts = step_text.split("meet ")
                 if len(parts) < 2: raise ValueError("Malformed meet step")
                 person = parts[1].split(" for")[0].strip()
                 if person not in constraints_info.get('person_constraints', {}):
                      raise ValueError(f"Person '{person}' mentioned in plan but not in constraints.")
                 meeting_duration = constraints_info['person_constraints'][person]['meeting_time']
                 cur_time += datetime.timedelta(minutes=meeting_duration)
            else:
                raise ValueError(f"Unknown plan step format: '{step_text[:50]}...'")
            plan_history.append(step_text)
        except ValueError as e:
            logging.error(f"[{example_id}] Error processing step '{step_text}': {e}. Stopping generation.")
            break
        except KeyError as e:
             logging.error(f"[{example_id}] Missing key during step processing '{step_text}': {e}. Stopping generation.")
             break
    return synthetic_data

def generate_cot_plan_with_gemini(
    gemini_model: GenerativeModel,
    problem_prompt: str,
    example_id: str
) -> str:
    """
    Uses the provided Gemini model to generate a CoT reasoning or plan.
    """
    logging.info(f"[{example_id}] Generating CoT plan with Gemini...")

    # Construct the prompt for Gemini, asking for step-by-step reasoning
    cot_prompt = (
        f"Given the following meeting planning problem, please provide a detailed step-by-step reasoning process and the final plan to satisfy the constraints. Think carefully about travel times, meeting durations, and availability windows.\n\n"
        f"Problem:\n{problem_prompt}\n\n"
        f"Reasoning and Plan:"
    )

    try:
        # Configure generation parameters for CoT
        generation_config = {
            "max_output_tokens": 2048,
            "temperature": 0.3, # Slightly higher temp for more detailed reasoning
            "top_p": 0.9,
            "top_k": 40
        }
        # Configure safety settings (optional, adjust as needed)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
        }

        response = gemini_model.generate_content(
            cot_prompt,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False, # Get the full response at once
        )

        # Extract text, handling potential empty or blocked responses
        if response and response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            generated_text = response.text.strip()
            logging.info(f"[{example_id}] Gemini CoT generation successful.")
            return generated_text
        else:
            finish_reason = response.candidates[0].finish_reason if response and response.candidates else "Unknown"
            safety_ratings = response.candidates[0].safety_ratings if response and response.candidates else "N/A"
            logging.warning(f"[{example_id}] Gemini response empty or blocked. Finish Reason: {finish_reason}, Safety: {safety_ratings}")
            return f"ERROR: Gemini response empty or blocked (Reason: {finish_reason})"

    except Exception as e:
        logging.error(f"[{example_id}] Error calling Gemini API: {e}")
        return f"ERROR: Exception during Gemini API call: {e}"


# --- Main Execution Logic ---

def main(argv: Sequence[str]) -> None:
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    gemini_model = None
    if FLAGS.use_gemini_cot:
        if not VERTEX_AI_AVAILABLE:
            logging.error("Vertex AI SDK is required for --use_gemini_cot=True but not installed.")
            return
        # Load environment variables for Vertex AI
        load_dotenv()
        PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
        LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION")
        if not PROJECT_ID or not LOCATION:
            logging.error("GOOGLE_CLOUD_PROJECT and GOOGLE_CLOUD_LOCATION env vars must be set for Gemini.")
            return
        try:
            logging.info(f"Initializing Vertex AI for project {PROJECT_ID} in {LOCATION}...")
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            gemini_model = GenerativeModel(FLAGS.gemini_model_name)
            logging.info(f"Using Gemini model: {FLAGS.gemini_model_name}")
        except Exception as e:
            logging.error(f"Failed to initialize Vertex AI or load Gemini model: {e}")
            return

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

    total_synthetic_records = 0
    examples_processed = 0

    try:
        with open(FLAGS.output_jsonl_path, 'w', encoding='utf-8') as outfile:
            for example_id, obj in input_data.items():
                if FLAGS.max_examples > 0 and examples_processed >= FLAGS.max_examples:
                    logging.info(f"Reached max examples limit ({FLAGS.max_examples}). Stopping.")
                    break
                examples_processed += 1
                logging.info(f"--- Processing Example: {example_id} ({examples_processed}/{len(input_data) if FLAGS.max_examples <= 0 else FLAGS.max_examples}) ---")

                if FLAGS.use_gemini_cot:
                    # --- Gemini CoT Generation Mode ---
                    problem_prompt_text = obj.get(FLAGS.prompt_key_for_gemini)
                    if not problem_prompt_text:
                        logging.warning(f"Skipping {example_id}: Missing prompt key '{FLAGS.prompt_key_for_gemini}'.")
                        continue

                    gemini_cot_output = generate_cot_plan_with_gemini(
                        gemini_model, problem_prompt_text, example_id
                    )

                    # Structure the output record for CoT mode
                    output_record = {
                        "id": example_id,
                        "problem_description": problem_prompt_text,
                        "gemini_cot_reasoning": gemini_cot_output
                    }
                    try:
                        outfile.write(json.dumps(output_record) + '\n')
                        total_synthetic_records += 1
                    except TypeError as e:
                         logging.error(f"[{example_id}] Error serializing CoT record to JSON: {e}")

                else:
                    # --- Golden Plan Decomposition Mode ---
                    raw_constraints = obj.get("constraints")
                    raw_dist_matrix = obj.get("dist_matrix")
                    raw_golden_plan = obj.get("golden_plan")

                    if not all([raw_constraints, raw_dist_matrix, raw_golden_plan]):
                        logging.warning(f"Skipping {example_id}: Missing fields for golden plan decomposition.")
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

                    synthetic_steps = generate_synthetic_steps_from_golden(
                        example_id, constraints_info, dist_matrix, golden_plan_steps
                    )

                    if synthetic_steps:
                        logging.info(f"Generated {len(synthetic_steps)} synthetic steps for {example_id}.")
                        for step_data in synthetic_steps:
                            try:
                                outfile.write(json.dumps(step_data) + '\n')
                                total_synthetic_records += 1
                            except TypeError as e:
                                 logging.error(f"[{example_id}] Error serializing step data to JSON: {e}")
                    else:
                        logging.warning(f"No synthetic steps generated for {example_id}.")

    except IOError as e:
        logging.error(f"Error writing to output file {FLAGS.output_jsonl_path}: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

    print("\n--- Synthetic Data Generation Summary ---")
    print(f"Mode: {'Gemini CoT' if FLAGS.use_gemini_cot else 'Golden Plan Decomposition'}")
    print(f"Processed {examples_processed} examples from input.")
    print(f"Generated a total of {total_synthetic_records} synthetic records/steps.")
    print(f"Output saved to: {FLAGS.output_jsonl_path}")


if __name__ == "__main__":
  app.run(main)
