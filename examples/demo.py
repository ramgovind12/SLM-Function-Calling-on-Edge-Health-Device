from healthfunc.model import HealthFunctionLM

model = HealthFunctionLM(
    repo_id="your-username/your-model",
    filename="your-model.gguf"
)

response = model.query("How was my sleep last week?")
print(response)