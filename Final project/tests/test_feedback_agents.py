import json
import unittest
from unittest.mock import patch

from feedback_agents import FeedbackClassifierAgent, FeedbackProcessingPipeline, TicketCreatorAgent


class FeedbackAgentsTests(unittest.TestCase):
    def setUp(self):
        self.classifier = FeedbackClassifierAgent(use_llm=True, api_key="test-key", model="test-model")
        self.ticket_creator = TicketCreatorAgent()
        self.pipeline = FeedbackProcessingPipeline(use_llm=True, api_key="test-key", model="test-model")

    def test_bug_classification_and_ticket_creation(self):
        fake_response = {
            "choices": [{"message": {"content": json.dumps({"category": "Bug", "confidence": 0.95})}}]
        }
        with patch("feedback_agents.urllib.request.urlopen") as mocked_urlopen:
            mocked_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(fake_response).encode("utf-8")
            category, confidence = self.classifier.classify("App crashes when I open settings", platform="App Store", rating=1)

        self.assertEqual(category, "Bug")
        self.assertGreaterEqual(confidence, 0.5)

        ticket = self.ticket_creator.create_ticket(
            source_id="rev_001",
            source_type="app_store_review",
            category=category,
            text="App crashes when I open settings",
            platform="App Store",
            app_version="3.0.1",
            rating=1,
            confidence=confidence,
        )
        self.assertEqual(ticket["category"], category)
        self.assertIn(ticket["priority"], {"Critical", "High", "Medium", "Low"})

    def test_pipeline_generates_outputs_from_csv_inputs(self):
        fake_response = {
            "choices": [{"message": {"content": json.dumps({"category": "Bug", "confidence": 0.95})}}]
        }
        with patch("feedback_agents.urllib.request.urlopen") as mocked_urlopen:
            mocked_urlopen.return_value.__enter__.return_value.read.return_value = json.dumps(fake_response).encode("utf-8")
            outputs = self.pipeline.process_all(
                data_dir="data",
                output_dir="output",
            )
        self.assertIn("generated_tickets.csv", outputs)
        self.assertIn("processing_log.csv", outputs)
        self.assertIn("metrics.csv", outputs)


if __name__ == "__main__":
    unittest.main()
