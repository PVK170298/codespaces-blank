import unittest

from feedback_agents import FeedbackClassifierAgent, FeedbackProcessingPipeline, TicketCreatorAgent


class FeedbackAgentsTests(unittest.TestCase):
    def setUp(self):
        self.classifier = FeedbackClassifierAgent()
        self.ticket_creator = TicketCreatorAgent()
        self.pipeline = FeedbackProcessingPipeline()

    def test_bug_classification_and_ticket_creation(self):
        text = "App crashes when I open settings after the latest update on iPhone 14"
        category, confidence = self.classifier.classify(text, platform="App Store", rating=1)
        self.assertEqual(category, "Bug")
        self.assertGreater(confidence, 0.5)

        ticket = self.ticket_creator.create_ticket(
            source_id="rev_001",
            source_type="app_store_review",
            category=category,
            text=text,
            platform="App Store",
            app_version="3.0.1",
            rating=1,
            confidence=confidence,
        )
        self.assertEqual(ticket["category"], "Bug")
        self.assertIn("crash", ticket["title"].lower())
        self.assertIn(ticket["priority"], {"Critical", "High", "Medium", "Low"})

    def test_llm_fallback_still_classifies_without_api_key(self):
        classifier = FeedbackClassifierAgent(use_llm=True)
        category, confidence = classifier.classify(
            "I cannot login after the latest update",
            platform="Google Play",
            rating=1,
        )
        self.assertEqual(category, "Bug")
        self.assertGreater(confidence, 0.5)

    def test_pipeline_generates_outputs(self):
        outputs = self.pipeline.process_all(
            data_dir="data",
            output_dir="output",
            overwrite=True,
        )
        self.assertIn("generated_tickets.csv", outputs)
        self.assertIn("processing_log.csv", outputs)
        self.assertIn("metrics.csv", outputs)


if __name__ == "__main__":
    unittest.main()
