

GATHER_INFORMATION_PROMPT = """
**Task:** Extract key travel planning details from the user's query below. Identify the following information if present:

* **Origin:** The starting city or location.
* **Destination:** The main destination city or location.
* **Number of People:** How many people are traveling.
* **Start Date:** The departure or start date of the trip.
* **End Date:** The return or end date of the trip.
* **Duration:** The length of the trip (e.g., number of days, weeks).
* **Budget:** The total budget amount and currency (if specified).
* **Preferences:** Any specific requests, interests, or constraints mentioned (e.g., "likes museums", "prefers budget hotels", "needs accessible options", "avoid flights before 9 am").

**Output Format:** Present the extracted information clearly, using key-value pairs. If a piece of information is not found in the query, explicitly state "Not specified".


**Extracted Constraints:**

Here is users query:
"""
