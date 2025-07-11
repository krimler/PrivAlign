import random
import numpy as np

# --- Dummy Local Model ---
class DummyLocalModel:
    def predict(self, prompt):
        return {
            "text": "Local response: " + prompt,
            "top_tokens": [("safe", 0.6), ("maybe", 0.3), ("risky", 0.1)]
        }

# --- Dummy Mentor ---
class DummyMentor:
    def __init__(self, name, epsilon):
        self.name = name
        self.epsilon = epsilon

    def query(self, prompt):
        return f"Mentor({self.name}) reply: [answering obfuscated] {prompt}"

# --- Domain Transfer Functions ---
def domain_transfer_function(prompt):
    substitutions = {
        "patient": "personX",
        "doctor": "advisor",
        "diagnosis": "condition",
        "prescription": "recommendation"
    }
    for k, v in substitutions.items():
        prompt = prompt.replace(k, v)
    return prompt

def reverse_transfer_function(prompt):
    substitutions = {
        "personX": "patient",
        "advisor": "doctor",
        "condition": "diagnosis",
        "recommendation": "prescription"
    }
    for k, v in substitutions.items():
        prompt = prompt.replace(k, v)
    return prompt

# --- PPO Reward and KPO Conversion ---
def reward_model(prompt, response):
    return len(response) / len(prompt)

def kpo_label_from_rewards(r1, r2, beta=1.0):
    exp1, exp2 = np.exp(beta * r1), np.exp(beta * r2)
    total = exp1 + exp2
    return exp1 / total, exp2 / total

def ppo_approx_response(prompt, candidates):
    scored = [(c, reward_model(prompt, c)) for c in candidates]
    return max(scored, key=lambda x: x[1])[0]

# --- Mentee Agent ---
class Mentee:
    def __init__(self, model, transfer_fn, reverse_fn, mentors, epsilon_query=0.5, threshold=0.75):
        self.model = model
        self.T = transfer_fn
        self.T_inv = reverse_fn
        self.mentors = mentors
        self.epsilon_query = epsilon_query
        self.threshold = threshold
        self.budget_used = 0.0

    def confidence_score(self, prompt):
        output = self.model.predict(prompt)
        probs = [p for _, p in output["top_tokens"]]
        entropy = -sum(p * np.log(p + 1e-6) for p in probs)
        return 1.0 - (entropy / np.log(len(probs)))

    def add_dp_noise(self, text, epsilon):
        chars = list(text)
        n = max(1, int(len(chars) / epsilon))
        for _ in range(n):
            idx = random.randint(0, len(chars) - 1)
            chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz ")
        return ''.join(chars)

    def query(self, prompt):
        score = self.confidence_score(prompt)
        if score >= self.threshold:
            return self.model.predict(prompt)["text"]

        # Escalation path with PPO → KPO → PPO
        x_prime = self.T(prompt)
        x_prime_noisy = self.add_dp_noise(x_prime, self.epsilon_query)

        # Simulate PPO-style reward responses
        responses = [
            "They should reduce salt intake.",
            "Consider ACE inhibitors under doctor supervision."
        ]
        r1 = reward_model(prompt, responses[0])
        r2 = reward_model(prompt, responses[1])
        p1, p2 = kpo_label_from_rewards(r1, r2, beta=2.0)
        print(f"KPO Labels (PPO → KPO): {p1:.2f}, {p2:.2f}")

        # Reverse KPO → PPO: select best response by reward again
        mentor_input = ppo_approx_response(prompt, responses)

        best_mentor = min(self.mentors.items(), key=lambda item: item[1].epsilon)
        mentor_name, mentor = best_mentor
        self.budget_used += self.epsilon_query + mentor.epsilon

        response = mentor.query(x_prime_noisy)
        decoded = self.T_inv(response)
        return f"[Mentor: {mentor_name}] {decoded}"

# --- Demo Execution ---
if __name__ == "__main__":
    local_llm = DummyLocalModel()
    mentors = {
        "mentorA": DummyMentor("A", 1.0),
        "mentorB": DummyMentor("B", 2.0),
    }

    mentee = Mentee(local_llm, domain_transfer_function, reverse_transfer_function, mentors)

    query = "What prescription is recommended for patient with recent diagnosis?"
    final_output = mentee.query(query)

    print("Final Output:", final_output)
    print("Privacy Budget Used:", mentee.budget_used)
