# SLM-Function-Calling-on-Edge-Health-Device
### Lightweight Function-Calling Language Models for Health Devices

This is a collection of lightweight language models fine-tuned to translate **natural language health queries into structured function calls** for health APIs and wearable devices.

The goal is to enable **privacy-first, on-device health assistants** that can run locally on **phones, wearables, and IoT devices** without requiring cloud inference.

Instead of manually navigating health dashboards, users can simply ask:

> "How has my metabolism been since I started my medication last week?"

The model interprets the request and converts it into a **structured API call** that applications can execute.

---

# 📦 Model Collection

This repository hosts multiple **fine-tuned lightweight models** optimized for health data function calling.

| Model | Base Model | Adapter Size | Format | Hugging Face |
|------|------------|-------------|--------|--------------|
| **HealthFunc-LM-L3** | Llama-3.2-3B | ~10MB | GGUF | https://huggingface.co/yourname/healthfunc-lm-l3 |

Model is optimized for:

- Function calling
- Temporal reasoning
- Health data APIs
- Edge inference

---

# Features

The models follow a simple reasoning pipeline:

1. Interpret user intent
2. Normalize temporal expressions
3. Generate structured function calls

Example reasoning block:

```html
<think>
User asks about metabolism trend.
"Last week" corresponds to a 7-day period.
Action: retrieve BMR data.
</think>
```

Output
```html
{
  "name": "get_bmr_data",
  "parameters": {
    "num_days": 7
  }
}
```

#  Recommended Format
```code
GGUF (4-bit quantized)
```

# Example

User
```code
How has my metabolism been since I started medication last week?
```

Code
```html
<think>
User is asking about metabolic trends.
"Last week" corresponds to a 7-day window.
Fetch BMR data.
</think>

<function>
{
  "name": "get_bmr_data",
  "parameters": {
    "num_days": 7
  }
}
</function>
```


