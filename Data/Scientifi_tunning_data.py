import os
from openai import OpenAI

# use LangGraph

client = OpenAI(
    api_key= '',
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

SYSTEM_PROMPT = "You are a Principal Python Algorithm Scientist."

USER_PROMPT_TEMPLATE = """
Please act as an Expert Algorithm Architect. I will provide you with the "Problem Context".

### Problem Context:
\"\"\"
{problem_context}
\"\"\"

Analyze the problem by strictly following these steps. DO NOT GENERATE ANY SOLUTION CODE. Be highly concise and dense with information.

### 1. Problem Deconstruction
* **Core Objective:** Summarize the ultimate goal in 1-2 precise sentences.
* **Constraints & Deductions:** Extract numerical boundaries and state 1 logical deduction (e.g., "N is up to 10^5, so O(N^2) will TLE; must aim for O(N log N)").
* **Example Insights:** Extract 1-2 critical rules, format requirements, or hidden traps revealed *specifically* by the Examples and Explanations.
* **Edge Cases:** Identify 2 extreme scenarios to test against (e.g., empty inputs, all identical elements).

### 2. Comprehensive I/O Specification
* **Data Structure:** Expected Input and Output types.
* **Mutation & State:** Does it require In-place modification? Is it a Stateful class design requiring internal variables?
* **Format & Ordering:** Are there strict rules on uniqueness, sorting, or valid return formats (e.g., "any order")?

### 3. Approach Exploration
Identify feasible approaches ranging from Brute Force to Optimal. For EACH, provide:
* **Approach Name:** (e.g., Sliding Window, Top-Down DP)
* **Core Logic:** A 2-sentence mechanical summary of how it works.
* **Complexity:** Time Complexity (Big O) & Space Complexity.

### 4. Optimal Selection
* **Best Approach:** State the single best approach for an interview.
* **Why:** 1 concise sentence justifying the choice based on constraints.

### 5. Test Plan (Dry-Run)
Provide 3 specific test cases to validate the Optimal approach. Format STRICTLY as:
* **Case 1 (Standard):** `In: [...]` -> `Expected: [...]`
* **Case 2 (Edge):** `In: [...]` -> `Expected: [...]`
* **Case 3 (Tricky/Mutation):** `In: [...]` -> `Expected: [...]`
(If in-place modification is required, "Expected" MUST show the final mutated state).
"""

Instruction = "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target. You may assume that each input would have exactly one solution, and you may not use the same element twice. You can return the answer in any order. Example 1: Input: nums = [2,7,11,15], target = 9 Output: [0,1] Explanation: Because nums[0] + nums[1] == 9, we return [0, 1]. Example 2: Input: nums = [3,2,4], target = 6 Output: [1,2] Example 3: Input: nums = [3,3], target = 6 Output: [0,1] Constraints: 2 <= nums.length <= 104 -109 <= nums[i] <= 109 -109 <= target <= 109 Only one valid answer exists. Follow-up: Can you come up with an algorithm that is less than O(n2) time complexity?"
Starter_code = "class Solution: def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:"
Instruction1 = "Given the root of a binary tree, determine if it is a valid binary search tree (BST). A valid BST is defined as follows: The left subtree of a node contains only nodes with keys less than the node's key. The right subtree of a node contains only nodes with keys greater than the node's key. Both the left and right subtrees must also be binary search trees. Example 1: Input: root = [2,1,3] Output: true Example 2: Input: root = [5,1,4,null,null,3,6] Output: false Explanation: The root node's value is 5 but its right child's value is 4. Constraints: The number of nodes in the tree is in the range [1, 104]. -231 <= Node.val <= 231 - 1"

final_user_prompt = USER_PROMPT_TEMPLATE.format(problem_context=Instruction1)
# |<Instruction_start>|...|<Instruction_end>||<Think>_start|...|<Think_end>||<Action_start>|...|<Action_end|>|<Observation_start>||<Observation_end>|
stream_response = client.chat.completions.create(
    model="qwen3.6-plus",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": final_user_prompt}
    ],
    stream=True,
    extra_body={"enable_thinking":True}
)


for chunk in stream_response:
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)

# the first approach think appending , hypothesize expected interpreter output with assert

"""
Based on context, provide 

Based on Problem Deconstruction, Comprehensive I/O Specification, and Approach Exploration. 
### 4. Optimal Selection
State the single best approach for an interview setting based on the constraints, and briefly explain why.
"""