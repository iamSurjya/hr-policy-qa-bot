from dataclasses import dataclass


SYSTEM_PROMPT = """You are an HR policy assistant for this company.
Your job is to answer employee questions accurately based strictly 
on the policy documents provided to you.

Rules you must follow:
- Only use information from the provided context to answer
- If the answer is not in the context, say exactly: 
  "I could not find that in the current policy documents. 
   Please contact HR directly for clarification."
- Never make up policies, numbers, dates, or entitlements
- If a question spans multiple policies, address each one clearly
- Keep answers concise and direct — employees want quick answers
- Always mention which policy document your answer comes from
"""


@dataclass
class PromptBuilder:
    """Builds the final prompt sent to the LLM.

    Keeping prompt construction in its own class means:
    - Easy to test (just check the output string)
    - Easy to iterate (change prompt without touching pipeline logic)
    - Single place to add things like conversation history later
    """

    system_prompt: str = SYSTEM_PROMPT

    def build(self, question: str, context: str) -> tuple[str, str]:
        """Returns (system, user_prompt) tuple.

        We return system and user separately because different LLM
        providers handle system prompts differently — Claude has a
        dedicated system parameter, Gemini prepends it to the prompt.
        Our providers.py handles that difference transparently.
        """
        user_prompt = f"""Use the following HR policy excerpts to answer 
the employee's question.

--- POLICY CONTEXT ---
{context}
--- END CONTEXT ---

Employee question: {question}

Answer:"""

        return self.system_prompt, user_prompt