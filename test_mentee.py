import unittest
from mentee import (
    DummyLocalModel, DummyMentor,
    domain_transfer_function, reverse_transfer_function,
    Mentee, MAX_QUERY_DEPTH
)

class TestMenteeAgent(unittest.TestCase):

    def setUp(self):
        self.local_llm = DummyLocalModel()
        self.mentors = {
            "mentorA": DummyMentor("A", 1.0),
            "mentorB": DummyMentor("B", 2.0),
        }
        self.mentee = Mentee(
            self.local_llm,
            domain_transfer_function,
            reverse_transfer_function,
            self.mentors,
            epsilon_query=0.5,
            threshold=0.99  # Force mentor path for testability
        )

    def test_local_confident(self):
        # Lower threshold to trigger local model
        self.mentee.threshold = 0.1
        response = self.mentee.query("simple query")
        self.assertTrue(response.startswith("Local response"))

    def test_mentor_used_when_confidence_low(self):
        response = self.mentee.query("query with doctor and prescription")
        self.assertIn("Reduce salt intake", response)  # PPO response from mentor

    def test_domain_transfer_roundtrip(self):
        raw = "The patient needs a diagnosis."
        transformed = domain_transfer_function(raw)
        recovered = reverse_transfer_function(transformed)
        self.assertIn("patient", recovered)
        self.assertIn("diagnosis", recovered)

    def test_dp_noise_nontrivial(self):
        noisy = self.mentee.add_dp_noise("This is a clean sentence.", epsilon=0.1)
        self.assertNotEqual(noisy, "This is a clean sentence.")

    def test_recursion_limit(self):
        self.mentee.threshold = 1.1  # Unreachable confidence
        with self.assertRaises(RuntimeError):
            self.mentee.query("always fail query", depth=MAX_QUERY_DEPTH + 1)

if __name__ == '__main__':
    unittest.main()
