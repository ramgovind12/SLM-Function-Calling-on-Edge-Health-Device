from llama_cpp import Llama
from healthfunc.model import HealthFunctionLM

llm = Llama.from_pretrained(
    repo_id="ramgovindv/mindcall_llama3_gguf",
    filename="Llama-3.2-3B-Instruct.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=8,
    verbose=False
)

model = HealthFunctionLM(
    repo_id="ramgovindv/mindcall_llama3_gguf",
    filename="Llama-3.2-3B-Instruct.Q4_K_M.gguf",
)


response2 = model.query("I've been eating a lot of fast food over the past two weeks and feel sluggish. Could this be related?")

print(response2)


import json
from tqdm import tqdm

def evaluate_model(model, tokenizer, test_dataset, max_new_tokens=128):
    model.eval()

    total = 0
    valid_json = 0
    correct_function = 0
    correct_arguments = 0
    exact_match = 0

    for sample in tqdm(test_dataset):
        prompt = sample["text"]
        ground_truth = sample["label"]

        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to("cuda")

        input_length = inputs.input_ids.shape[1]

        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            use_cache=False
        )

        generated = tokenizer.decode(
            outputs[0][input_length:], 
            skip_special_tokens=True
        ).strip()

        total += 1

        # --- Try parsing JSON ---
        try:
            pred_json = json.loads(generated)
            true_json = json.loads(ground_truth)
            valid_json += 1
        except:
            continue  # invalid JSON → skip deeper checks

        # --- Function name accuracy ---
        if pred_json.get("function_name") == true_json.get("function_name"):
            correct_function += 1

        # --- Argument accuracy ---
        if pred_json.get("arguments") == true_json.get("arguments"):
            correct_arguments += 1

        # --- Exact match ---
        if pred_json == true_json:
            exact_match += 1

    # Metrics
    results = {
        "total_samples": total,
        "json_validity_rate": valid_json / total,
        "function_accuracy": correct_function / total,
        "argument_accuracy": correct_arguments / total,
        "exact_match_accuracy": exact_match / total,
    }

    return results