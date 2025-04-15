from flask import Flask, request, jsonify, render_template
import openai  # Install with: pip install openai

app = Flask(__name__)


class DebatingAgent:
    def __init__(self, name, strategy):
        self.name = name
        self.strategy = strategy  # Strategy: "logical", "emotional"
        self.memory = []  # Stores previous arguments

    def act(self, topic, opponent_memory):
        try:
            prompt = f"""You are a debating agent named {self.name} who debates in a {self.strategy} style.
            The debate topic is: {topic}
            The opponent has argued: {opponent_memory}
            Respond concisely  using a {self.strategy} approach in no more than 50 words
            """


            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "system", "content": prompt}]
            )

            argument = response["choices"][0]["message"]["content"]
            self.memory.append(argument)
            return argument
        except Exception as e:
            return f"Error: {e}"

agent1 = DebatingAgent("Agent Logic", "logical")
agent2 = DebatingAgent("Agent Emotion", "emotional")

@app.route('/')
def home():
    return render_template('letsdebate.html')

@app.route('/start_debate', methods=['POST'])
def start_debate():
    data = request.json
    topic = data['topic']
    rounds = int(data['rounds'])

    agent1.memory = []
    agent2.memory = []
    debate_history = []

    for i in range(rounds):
        agent1_response = agent1.act(topic, agent2.memory)
        agent2_response = agent2.act(topic, agent1.memory)
        debate_history.append({
            "round": i + 1,
            "agent1": agent1_response,
            "agent2": agent2_response
        })

    return jsonify(debate_history)

if __name__ == '__main__':
    app.run(debug=True)

