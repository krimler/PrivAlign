import random
import numpy as np

# --- Constants ---
MAX_QUERY_DEPTH = 7

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

    def query(self, prompt, context=None):
        return f"Mentor({self.name}) reply to preference: {context} → {prompt}"

# --- Domain Transfer Functions ---
def domain_transfer_function(prompt):
    substitutions = {
        "patient": "personX",
        "doctor": "advisor",
        "diagnosis": "condition",
        "prescription": "recommendation",
    }
    for key, val in substitutions.items():
        prompt = prompt.replace(key, val)
    return prompt

def reverse_transfer_function(prompt):
    substitutions = {
        "personX": "patient",
        "advisor": "doctor",
        "condition": "diagnosis",
        "recommendation": "prescription",
    }
    for key, val in substitutions.items():
        prompt = prompt.replace(key, val)
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

    def query(self, prompt, depth=0):
        if depth > MAX_QUERY_DEPTH:
            raise RuntimeError("Failed to obtain high-confidence response within recursion limit.")

        score = self.confidence_score(prompt)
        if score >= self.threshold:
            return self.model.predict(prompt)["text"]

        return self._mentor_escalation(prompt, depth)

    def _mentor_escalation(self, prompt, depth):
        x_prime = self.T(prompt)
        x_prime_noisy = self.add_dp_noise(x_prime, self.epsilon_query)

        cand_a = "Reduce salt intake."
        cand_b = "Use ACE inhibitors under supervision."
        r_a = reward_model(prompt, cand_a)
        r_b = reward_model(prompt, cand_b)
        p_a, p_b = kpo_label_from_rewards(r_a, r_b, beta=2.0)
        chosen = ppo_approx_response(prompt, [cand_a, cand_b])

        preference_context = (
            f"Preference between two responses:
"
            f"A: {cand_a}
"
            f"B: {cand_b}
"
            f"Soft preference: A={p_a:.2f}, B={p_b:.2f}
"
            f"Preferred direction: {'A' if r_a > r_b else 'B'}"
        )

        best_mentor = min(self.mentors.items(), key=lambda m: m[1].epsilon)[1]
        self.budget_used += self.epsilon_query + best_mentor.epsilon

        y_prime = best_mentor.query(x_prime_noisy, context=preference_context)
        y = self.T_inv(y_prime)

        post_noise_score = self.confidence_score(y)
        if post_noise_score < self.threshold:
            return self.query(prompt, depth=depth + 1)

        reward = reward_model(prompt, y)
        print(f"Recovered PPO reward for mentor response: {reward:.3f}")

        return ppo_approx_response(prompt, [y, chosen])

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
