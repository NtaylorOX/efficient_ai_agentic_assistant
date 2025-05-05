# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Reviser agent for correcting inaccuracies based on verified findings."""

from google.adk import Agent
from . import prompt


planner_agent = Agent(
    # model=LiteLlm("openai/gpt-4o"),
    model='gemini-2.0-flash',
    name='planner_agent',
    instruction=prompt.COT_PLANNER_INSTRUCTION,
    description="An agent that breaks down a complex task into smaller, manageable steps. It provides a step-by-step plan to achieve the goal.",
    output_key="step_by_step_plan"
)
