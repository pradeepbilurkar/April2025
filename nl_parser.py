import openai
import json

def interpret_layout_query(user_prompt, api_key, model="gpt-4"):
    openai.api_key = api_key

    system_prompt = """You are a warehouse layout assistant. Given a user request, extract these parameters in compact JSON:
{
  "truck_count": int,
  "truck_type": "short" | "medium" | "long",
  "clearance": float,
  "optimize_goal": "maximize_trucks" | "minimize_area",
  "lane_width": float
}

Use reasonable defaults if values are vague. Only reply with JSON, no explanations."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages
        )
        content = response["choices"][0]["message"]["content"].strip()

        # Try parsing JSON
        params = json.loads(content)
        return params

    except Exception as e:
        print("❌ NLP parsing failed:", str(e))
        return None