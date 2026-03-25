import os
import logging
import warnings

# 1. Silence Python UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Silence Hugging Face Hub (removes the HF_TOKEN warning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# 3. Silence llama-cpp-python's internal C++ logs (removes n_ctx_seq messages)
os.environ["GGML_PYTHON_VERBOSE"] = "0" 
os.environ["LLAMA_CPP_LIB_VERBOSE"] = "0"


from llama_cpp import Llama
import json
class HealthFunctionLM:
    def __init__(self, repo_id, filename, n_ctx=2048, n_threads=4):
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            n_threads=n_threads,
            chat_format=None,
            verbose=False
        )

    def _build_prompt(self, query):
        return f"""
    You are an API generator.

    Convert the user query into a JSON function call.

    Rules:
    - Output ONLY valid JSON
    - No explanation
    - No text outside JSON

    Available function:
    get_bmr_data(num_days: int)

    User query:
    {query}

    JSON:
"""

    def _generate(self, query):
            # 1. Use your prompt builder to tell the model about the rules and tools
            full_prompt = self._build_prompt(query)

            # 2. Pass that prompt as the 'content'
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "user", "content": full_prompt}
                ],
                # If your model supports native tool-calling, uncomment this:
                # tools=self.tools, 
                temperature=0.1
            )

            return response

    def query(self, user_query):
        response = self._generate(user_query)
        message = response["choices"][0]["message"]

        # Case A: Model used the native tool_calls structure
        if "tool_calls" in message and message["tool_calls"]:
            return {
                "query": user_query,
                "function_call": message["tool_calls"][0]["function"]
            }

        # Case B: Model returned JSON as a string in 'content'
        content = message.get("content", "").strip()
        if content:
            try:
                # Attempt to parse the string as JSON
                parsed_json = json.loads(content)
                return {
                    "query": user_query,
                    "function_call": parsed_json
                }
            except json.JSONDecodeError:
                # Not JSON, return as plain text
                pass

        return {
            "query": user_query,
            "response": content
        }