import json
import re 

def extract_fact_checking_results(response_text: str) -> dict:
    """
    Extracts the fact-checking results (issues, and scores) from a JSON-formatted string,
    robustly handling markdown code block delimiters (```json ... ```) and surrounding text/whitespace.

    Args:
        response_text: The string possibly containing the JSON-formatted fact-checking results.

    Returns:
        A dictionary containing the extracted information, or a default structure in case of errors.
    """
    default_result = {
        "issues": [],
        "overall_accuracy_score": None,
        "target_audience_score": None,
        "relevance_score": None
    }
    print(f"DEBUG: Raw input to parser: '{response_text[:200]}...'") # Optional: Debug input

    json_string = None

    # Use regex to find content within ```json ... ```, allowing for whitespace and newlines
    # re.DOTALL makes '.' match newlines as well
    match = re.search(r"```json\s*(\{.*?\})\s*```", response_text, re.DOTALL)

    if match:
        json_string = match.group(1).strip() # Extract the captured group (the JSON object) and strip whitespace
        print(f"DEBUG: Extracted JSON string via regex: '{json_string}'")
    else:
        # Fallback: If no markdown fences found, maybe the response *is* just raw JSON?
        # Trim whitespace aggressively and check if it looks like a JSON object.
        trimmed_response = response_text.strip()
        if trimmed_response.startswith("{") and trimmed_response.endswith("}"):
             json_string = trimmed_response
             print(f"DEBUG: Assuming raw JSON string after trim: '{json_string}'")
        else:
             # If it's not in fences and doesn't look like raw JSON, give up.
             print(f"DEBUG: Could not find JSON block via regex or identify raw JSON object.")
             return default_result

    # Proceed only if we potentially have a JSON string
    if json_string:
        try:
            # Attempt to parse the extracted/cleaned JSON string
            data = json.loads(json_string)

            # Validate the structure of the parsed data
            if isinstance(data, dict):
                issues = data.get("issues")
                accuracy_score = data.get("overall_accuracy_score")
                target_score = data.get("target_audience_score")
                relevance_score = data.get("relevance_score")

                # Basic type checking (adjust if scores can be floats)
                if (
                    isinstance(issues, list)
                    # and all(isinstance(issue, str) for issue in issues) # Can relax this if issues can be complex
                    and (accuracy_score is None or isinstance(accuracy_score, int)) # Or (int, float)
                    and (target_score is None or isinstance(target_score, int))    # Or (int, float)
                    and (relevance_score is None or isinstance(relevance_score, int)) # Or (int, float)
                ):
                    # Ensure keys exist even if None, using .get with default None
                    return {
                        "issues": data.get("issues", []), # Default to empty list if missing
                        "overall_accuracy_score": data.get("overall_accuracy_score"),
                        "target_audience_score": data.get("target_audience_score"),
                        "relevance_score": data.get("relevance_score")
                    }
                else:
                    print(f"Error: Invalid data types within JSON object. Data: {data}")
                    return default_result
            else:
                print(f"Error: Expected a JSON object (dict) but found type {type(data)}.")
                return default_result

        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON format in extracted string '{json_string}': {e}")
            return default_result
        except Exception as e:
            print(f"An unexpected error occurred during parsing or validation: {e}")
            return default_result
    else:
         # Should have been caught earlier, but as a safeguard:
         print("Error: No valid JSON string could be extracted.")
         return default_result


def save_results_to_file(results: dict, filename: str = "fact_check_results.json") -> None:
    """
    Saves the fact-checking results (dictionary) to a JSON file.

    Args:
        results: A dictionary containing the fact-checking results (as returned by extract_fact_checking_results).
        filename: The name of the file to save the results to (default: "fact_check_results.json").
    """
    try:
        with open(filename, "w") as f:
            json.dump(results, f, indent=4)  # Use indent for pretty formatting
        print(f"Successfully saved results to {filename}")
    except Exception as e:
        print(f"Error saving results to file: {e}")



# if __name__ == "__main__":
#     # Example usage with a valid JSON response
#     valid_response = '{"issues": ["Issue 1: Claim X is inaccurate because of Y.", "Issue 2: Claim Z lacks sufficient evidence."], "overall_accuracy_score": 6}'
#     results = extract_fact_checking_results(valid_response)
#     print("Extracted results (valid):", results)
#     save_results_to_file(results, "valid_results.json")

#     # Example usage with an invalid JSON response (missing score)
#     invalid_response_missing_score = '{"issues": ["Issue 1: ..."]}'
#     results = extract_fact_checking_results(invalid_response_missing_score)
#     print("Extracted results (missing score):", results)
#     save_results_to_file(results, "missing_score_results.json")

#     # Example usage with an invalid JSON response (invalid score type)
#     invalid_response_wrong_score_type = '{"issues": [], "overall_accuracy_score": "not a number"}'
#     results = extract_fact_checking_results(invalid_response_wrong_score_type)
#     print("Extracted results (wrong score type):", results)
#     save_results_to_file(results, "wrong_score_type.json")

#     # Example usage with a completely invalid JSON string
#     invalid_json_string = "This is not valid JSON"
#     results = extract_fact_checking_results(invalid_json_string)
#     print("Extracted results (invalid JSON):", results)
#     save_results_to_file(results, "invalid_json.json")
