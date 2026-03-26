import os
import logging
import warnings
from llama_cpp import Llama
import json
import re

# 1. Silence Python UserWarnings
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Silence Hugging Face Hub (removes the HF_TOKEN warning)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# 3. Silence llama-cpp-python's internal C++ logs (removes n_ctx_seq messages)
os.environ["GGML_PYTHON_VERBOSE"] = "0" 
os.environ["LLAMA_CPP_LIB_VERBOSE"] = "0"


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

    def _build_prompt(self,query):
        return f"""
    You are an API generator.

    Return JSON in this format:

    {{
    "name": "function_name",
    "parameters": {{
        "key": "value"
    }}
    }}

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
    
    def extract_think(self,text):
        pattern = r"<think>\s*(.*?)\s*</think>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def extract_function_call(self,text):
        pattern = r"<function>\s*(\{.*?\})\s*</function>"
        match = re.search(pattern, text, re.DOTALL)

        if not match:
            return None

        try:
            return json.loads(match.group(1))
        except:
            return None

    def query(self, user_query):
        response = self._generate(user_query)
        message = response["choices"][0]["message"]

        # Case A: tool_calls
        if "tool_calls" in message and message["tool_calls"]:
            fc = message["tool_calls"][0]["function"]

            return {
                "query": user_query,
                "type": "function_call",
                "data": {
                    "name": fc.get("name"),
                    "parameters": fc.get("arguments", {}),
                    "reasoning": None  # tool_calls usually don't include reasoning
                }
            }

        content = message.get("content", "").strip()

        if content:
            reasoning = self.extract_think(content)
            function_data = self.extract_function_call(content)

            if function_data:
                return {
                    "query": user_query,
                    "type": "function_call",
                    "data": {
                        "name": function_data.get("name"),
                        "parameters": function_data.get("parameters", {}),
                        "reasoning": reasoning
                    }
                }

            # fallback: raw JSON (no tags)
            try:
                parsed_json = json.loads(content)
                return {
                    "query": user_query,
                    "type": "function_call",
                    "data": {
                        "name": parsed_json.get("name"),
                        "parameters": parsed_json.get("parameters", {}),
                        "reasoning": reasoning
                    }
                }
            except:
                pass

        return {
            "query": user_query,
            "type": "text",
            "data": {
                "content": content,
                "reasoning": None
            }
        }