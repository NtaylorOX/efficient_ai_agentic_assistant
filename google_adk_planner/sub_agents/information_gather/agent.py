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

"""Scientific Writer Agent. Focused on writing scientific reports for target audiences."""

from google.adk import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.genai import types
from google.adk.models.lite_llm import LiteLlm
from . import prompt

# create web search agent
information_gather_agent = Agent(
    # model=LiteLlm("openai/gpt-4o"),
    model='gemini-2.0-flash',
    name='information_gather_agent',
    description="This agent is responsible for gathering information and constraints from the user. It will ask questions to clarify the user's needs and gather relevant information.",
    instruction=prompt.GATHER_INFORMATION_PROMPT,
    output_key="information_and_constraints",
)