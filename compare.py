# Simulated Federated RLHF Training Pipeline (toy example)
import random
import numpy as np

# Dummy Local Client
class RLHFClient:
    def __init__(self, name):
        self.name = name
        self.data = []
        self.rewards = []

    def generate_data(self):
        prompts = ["What is hypertension treatment?", "How to reduce blood pressure?"]
        responses = [
            "Use ACE inhibitors.",
            "Lower salt intake.",
            "Exercise regularly.",
        ]
        self.data = [(p, random.choice(responses)) for p in prompts]

    def assign_rewards(self):
        self.rewards = [(p, r, self.add_dp_noise(self.reward_model(p, r))) for p, r in self.data]

    def add_dp_noise(self, score, epsilon=0.5):
        noise = np.random.laplace(loc=0.0, scale=1/epsilon)
        return score + noise

    (self, prompt, response):
        return len(response) / len(prompt)

    def get_preference_pairs(self):
        prefs = []
        for (p1, r1, s1) in self.rewards:
            for (p2, r2, s2) in self.rewards:
                if p1 == p2 and s1 != s2:
                    preferred = (r1, r2) if s1 > s2 else (r2, r1)
                    prefs.append((p1, preferred[0], preferred[1]))
        return prefs

# Federated Aggregator
class RLHFAggregator:
    def __init__(self, clients):
        self.clients = clients
        self.global_prefs = []

    def collect_preferences(self):
        all_pairs = []
        for client in self.clients:
            client.generate_data()
            client.assign_rewards()
            pairs = client.get_preference_pairs()
            all_pairs.extend(pairs)
        self.global_prefs = all_pairs

    def train_global_model(self):
        print("[Aggregator] Training on", len(self.global_prefs), "preference pairs")
        mentee_sample = ("What prescription is recommended for patient with recent diagnosis?", "Reduce salt intake.", "Use ACE inhibitors under supervision.")
        r1 = len(mentee_sample[1]) / len(mentee_sample[0])
        r2 = len(mentee_sample[2]) / len(mentee_sample[0])
        beta = 2.0
        p1 = np.exp(beta * r1) / (np.exp(beta * r1) + np.exp(beta * r2))
        p2 = 1 - p1
        print(f"
[Compare to Mentee PPO→KPO Conversion]
Prompt: {mentee_sample[0]}
Response A: {mentee_sample[1]}
Response B: {mentee_sample[2]}
Soft Preference → A: {p1:.2f}, B: {p2:.2f}
")

        # Agreement count
        mentee_preference = mentee_sample[1] if p1 > p2 else mentee_sample[2]
        match = 0
        total = 0
        for _, win, _ in self.global_prefs:
            if win == mentee_preference:
                match += 1
            total += 1
        agreement_rate = match / total if total > 0 else 0.0
        print(f"[Agreement Rate with Mentee Preference]: {agreement_rate:.2f}
")

        return self.global_prefs

# --- Comparison Logic: Mentee vs Federated RLHF ---
def compare_to_mentee_vs_federated(mentee_prompt, mentee_response_A, mentee_response_B):
    r1 = len(mentee_response_A) / len(mentee_prompt)
    r2 = len(mentee_response_B) / len(mentee_prompt)
    beta = 2.0
    p1 = np.exp(beta * r1) / (np.exp(beta * r1) + np.exp(beta * r2))
    p2 = 1 - p1
    print("
[Mentee PPO→KPO Soft Preference Conversion]")
    print(f"Prompt: {mentee_prompt}")
    print(f"Response A: {mentee_response_A}")
    print(f"Response B: {mentee_response_B}")
    print(f"Soft Preference → A: {p1:.2f}, B: {p2:.2f}
")

    print("[Federated RLHF Preference Agreement Example]")
    print("(Showing top preferences from federated clients with DP noise)")

# --- Token Cost Comparison ---
def simulate_token_cost(prompt, response, method):
    base_tokens = len(prompt.split()) + len(response.split())
    if method == "mentee":
        # Domain-transferred prompt is simpler; assume 40% reduction in prompt size
        kpo_prompt_tokens = int(len(prompt.split()) * 0.6)
        kpo_response_tokens = len(response.split())
        total = kpo_prompt_tokens + kpo_response_tokens
        return total
    elif method == "federated":
        return base_tokens * 0.25

# --- DP-SFT Simulation (baseline) ---
class DPSFTClient:
    def __init__(self, epsilon=0.5):
        self.epsilon = epsilon

    def reward(self, prompt, response):
        base_score = len(response) / len(prompt)
        noise = np.random.laplace(0.0, 1 / self.epsilon)
        return base_score + noise

    def soft_preference(self, prompt, resp_a, resp_b):
        s1 = self.reward(prompt, resp_a)
        s2 = self.reward(prompt, resp_b)
        beta = 2.0
        p1 = np.exp(beta * s1) / (np.exp(beta * s1) + np.exp(beta * s2))
        return p1, 1 - p1

def compare_to_dp_sft(mentee_prompt, mentee_response_A, mentee_response_B):
    client = DPSFTClient(epsilon=0.5)
    p1, p2 = client.soft_preference(mentee_prompt, mentee_response_A, mentee_response_B)
    print("
[DP-SFT Soft Preference (ε=0.5)]")
    print(f"Prompt: {mentee_prompt}")
    print(f"Response A: {mentee_response_A}")
    print(f"Response B: {mentee_response_B}")
    print(f"Soft Preference → A: {p1:.2f}, B: {p2:.2f}
")

    mentee_pref = mentee_response_A if len(mentee_response_A) > len(mentee_response_B) else mentee_response_B
    dp_sft_pref = mentee_response_A if p1 > p2 else mentee_response_B
    agreement = mentee_pref == dp_sft_pref
    print(f"[Agreement with Mentee Preference]: {agreement}
")
if __name__ == "__main__":
    clients = [RLHFClient(f"Client{i}") for i in range(3)]
    aggregator = RLHFAggregator(clients)

    compare_to_dp_sft(mentee_prompt, mentee_response_A, mentee_response_B)

    # Compare mentee preference to federated
    mentee_prompt = "What prescription is recommended for patient with recent diagnosis?"
    mentee_response_A = "Reduce salt intake."
    mentee_response_B = "Use ACE inhibitors under supervision."
    compare_to_mentee_vs_federated(mentee_prompt, mentee_response_A, mentee_response_B)

    aggregator.collect_preferences()
    trained_pairs = aggregator.train_global_model()

    mentee_tokens = simulate_token_cost(mentee_prompt, mentee_response_A, method="mentee")
    federated_tokens = simulate_token_cost(mentee_prompt, mentee_response_A, method="federated")
    print(f"
[Token Cost per Query Resolution]
Mentee Token Cost: {mentee_tokens:.1f}
Federated Token Cost (per client, amortized): {federated_tokens:.1f}
")

    for i, (prompt, win, lose) in enumerate(trained_pairs[:5]):
        conflict = win not in [mentee_response_A, mentee_response_B]
        print(f"Example {i}: Prompt='{prompt}' | Preferred: '{win}' > '{lose}' | Conflict with mentee={conflict}")
