Developing AI Planner model(s) for an AI Assistant 

Overview 

We are aiming to develop an AI assistant that requires a specialized AI planner model(s). The AI planner will be used to guide the assistant's actions and decisions, making it crucial for it to be efficient and effective. Given the constraints of running this system on edge devices such as mobile phones, we need to consider the size and complexity of the model. 

Task Description 

Your task is to develop a strategy for training an AI planner model(s) specifically for our AI assistant. The model(s) must be designed to work efficiently within the constraints of edge devices. The model(s) must be 8B parameters or smaller. 

The model(s) should be able to handle different tasks such as: 

\- Planning a trip: e.g., "Plan a trip to Europe for two weeks". Constrained by budget, days, weather, preferences, etc. 

\- Meeting planning: e.g., "Arrange for a meeting with John”. Constrained by weather, calendar availability, transit time, reservation availability, etc. 

\- etc. 

The performance should be comparable to that of a large-scale AI planner, while still running on a device that can handle only a 8B model. 

Deliverables 

**1\. Strategy Document:** A detailed document outlining your proposed strategy, including: \- The architecture and specifications of the proposed AI planner model(s). \- The training and evaluation strategy for the model(s). 

\- Methods for optimizing the model(s) for edge devices. 

\- Techniques for integrating the AI planning model(s) into the AI assistant. 

**2\. Code examples**: 

\- Training the planning model(s). 

\- Assistant integration. 

\- Model evaluation.  
Resources 

\- You may use https://arxiv.org/pdf/2406.04520 as a benchmark for the evaluation of a planner. \- You may consider https://github.com/mlc-ai/mlc-llm for edge inference. 

Task evaluation criteria 

\- Performance: Potential performance of the model(s) based on proposed training \- Feasibility: The model(s) should be small enough for running on edge devices. \- Integration: The AI planner should be seamlessly integrated into the AI assistant. \- Efficiency: Optimization of the model(s) for performance for edge devices. \- Innovativeness: Creativity and originality of the proposed solution. 

We understand the scope of the task is quite large. Please accomplish the maximum you are able to within the given timeframe. While everything needs to be addressed at least at a high level, you could go in depth into areas that are more critical / innovation is needed. 

We are also aware that many aspects of the task, especially regarding the model(s) training and evaluation, are left open. Please make reasonable choices to best fulfill on the job. 

Submission 

**The deadline to return your solution to us is 5 days from the date of receipt. Please send the strategy document as a PDF and code samples as Zip file over email or as a private Github repo shared with `subashsn`**