import openai

def narrate_layout_feedback(feedback_summary, api_key, model="gpt-4"):
    openai.api_key = "A"

    prompt = f"""You're a warehouse layout expert. Here's a diagnostic summary of truck placements:
{feedback_summary}

Generate a readable narrative highlighting the layout issues, safety concerns, and improvement suggestions."""

    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a logistics layout expert."},
                {"role": "user", "content": prompt}
            ]
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ AI narration failed: {str(e)}"