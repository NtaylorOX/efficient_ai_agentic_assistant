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

"""Task planning agent"""

from google.adk.agents import SequentialAgent, LoopAgent, Agent
from .sub_agents.planner import planner_agent
from .sub_agents.information_gather import information_gather_agent


task_planner_agent = SequentialAgent(
    name='task_planner',
    description=(
        'An agent that breaks down a complex task into smaller, manageable steps.'
        ' It provides a step-by-step plan to achieve the goal.'
        'First uses the information gathering agent to collect relevant information,'
        ' then uses the planning agent to create a detailed plan.'
    ),
    sub_agents=[
        information_gather_agent,
        planner_agent
    ]
)


root_agent = task_planner_agent
