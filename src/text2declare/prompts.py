from __future__ import annotations

# Prompt used for supervised fine-tuning (SFT)
PROMPT_QA = (
    "### Question: What is the Declare constraint described in the following sentence? {description}\n"
    "### Answer: {constraint}"
)

# Few-shot prompt for direct inference (prompting baselines)
FEW_SHOT_PROMPT = (
    """I am a highly intelligent AI capable of understanding and translating natural language descriptions into Declare constraints. Here are some example translations of descriptions into Declare constraints:
1. Description: Breakfast must be the first to occur.
Task(s): Breakfast
Declare Constraint: Init(Breakfast)
*more shots*
11. Description: Cooking and Biking never occur together.
Task(s): Cooking, Biking
Declare Constraint: NotCoExistence(Cooking, Biking)
Given the following description, we should first detect tasks and then translate the description into a Declare constraint. The output format should exactly follow this pattern as in the example descriptions. The answer should solely contain the constraint and nothing else, i.e., no additional description. Now, let's translate the following description into a Declare constraint:
12. Description: {description}
"""
)

# Few-shot + CoT prompt for direct inference
FEW_SHOT_COT_PROMPT_TEMPLATE = (
    """<start_of_turn>user
You are a highly intelligent AI capable of understanding and translating natural language descriptions into Declare constraints. Here are definitions of some common Declare templates:
1. Definition: "Task A must be the first to occur."
Declare: Init(A)
*more definitions*
11. Definition: "Task A and Task B never occur together."
Declare: NotCoExistence(A, B)
Think and write your step-by-step reasoning before responding.
Now, translate the following description into a Declare constraint: "{description}"
<end_of_turn>
<start_of_turn>model
Let's think step by step.
"""
)
