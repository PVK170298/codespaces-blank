import csv
import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(dotenv_path=BASE_DIR / ".env")


@dataclass
class FeedbackRecord:
    source_id: str
    source_type: str
    text: str
    platform: str = ""
    rating: int = 0
    app_version: str = ""
    subject: str = ""
    sender_email: str = ""
    timestamp: str = ""
    priority: str = ""


class CSVReaderAgent:
    def read_reviews(self, file_path: str) -> List[FeedbackRecord]:
        df = pd.read_csv(file_path)
        records = []
        for _, row in df.iterrows():
            records.append(
                FeedbackRecord(
                    source_id=str(row.get("review_id", "")),
                    source_type="app_store_review",
                    text=str(row.get("review_text", "")),
                    platform=str(row.get("platform", "")),
                    rating=int(row.get("rating", 0)),
                    app_version=str(row.get("app_version", "")),
                )
            )
        return records

    def read_support_emails(self, file_path: str) -> List[FeedbackRecord]:
        df = pd.read_csv(file_path)
        records = []
        for _, row in df.iterrows():
            records.append(
                FeedbackRecord(
                    source_id=str(row.get("email_id", "")),
                    source_type="support_email",
                    text=f"{row.get('subject', '')} {row.get('body', '')}",
                    subject=str(row.get("subject", "")),
                    sender_email=str(row.get("sender_email", "")),
                    timestamp=str(row.get("timestamp", "")),
                    priority=str(row.get("priority", "")),
                )
            )
        return records


class FeedbackClassifierAgent:
    def __init__(self, use_llm: bool = False, api_key: Optional[str] = None, model: Optional[str] = None, api_base: Optional[str] = None):
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.class_names = ["Bug", "Feature Request", "Praise", "Complaint", "Spam"]
        self.reference_texts = [
            "app crashes bug error fails login issue cannot access sync not working",
            "please add feature request would love to see dark mode missing functionality",
            "love amazing great works perfectly fantastic helpful",
            "bad slow expensive poor customer service frustrating disappointing",
            "promo buy now click here random unrelated spam",
        ]
        self.reference_vectors = self.vectorizer.fit_transform(self.reference_texts)
        self.use_llm = use_llm
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", os.getenv("OPENAI_API_KEY", ""))
        self.model = model or os.getenv("GEMINI_MODEL", os.getenv("OPENAI_MODEL", "gemini-2.0-flash"))
        self.api_base = (api_base or os.getenv("GEMINI_API_BASE", os.getenv("OPENAI_API_BASE", "https://generativelanguage.googleapis.com/v1beta/models"))).rstrip("/")
        self.last_llm_used = False

    def classify(self, text: str, platform: str = "", rating: int = 0) -> Tuple[str, float]:
        self.last_llm_used = False
        lowered = (text or "").lower()
        if any(term in lowered for term in ["promo", "buy now", "click here", "random", "xxx", "spam"]):
            return "Spam", 0.97
        if any(term in lowered for term in ["crash", "login", "sync", "error", "bug", "issue", "not working", "data loss"]):
            return "Bug", 0.93
        if any(term in lowered for term in ["please add", "feature request", "would love", "missing functionality", "dark mode", "suggestion"]):
            return "Feature Request", 0.91
        if any(term in lowered for term in ["love", "amazing", "great", "perfectly", "fantastic", "awesome"]):
            return "Praise", 0.9
        if any(term in lowered for term in ["expensive", "slow", "poor", "frustrating", "terrible", "bad"]):
            return "Complaint", 0.88

        if self.use_llm and self.api_key:
            llm_result = self._classify_with_llm(text, platform, rating)
            if llm_result:
                self.last_llm_used = True
                return llm_result

        vector = self.vectorizer.transform([text])
        similarities = cosine_similarity(vector, self.reference_vectors).flatten()
        best_idx = int(similarities.argmax())
        best_score = float(similarities[best_idx])
        category = self.class_names[best_idx]
        return category, max(best_score, 0.3)

    def _classify_with_llm(self, text: str, platform: str = "", rating: int = 0) -> Optional[Tuple[str, float]]:
        prompt = (
            "Classify the following user feedback into exactly one of these categories: "
            "Bug, Feature Request, Praise, Complaint, Spam. "
            "Return JSON with keys 'category' and 'confidence'. "
            f"Feedback: {text} | platform: {platform or 'unknown'} | rating: {rating}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": "You are a support ticket triage assistant. " + prompt}
                    ]
                }
            ],
            "generationConfig": {"temperature": 0},
        }
        headers = {
            "Content-Type": "application/json",
        }
        try:
            endpoint = f"{self.api_base}/{self.model}:generateContent?key={self.api_key}"
            req = urllib.request.Request(
                endpoint,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as response:
                data = json.load(response)
            content = data["candidates"][0]["content"]["parts"][0]["text"]
            parsed = json.loads(content)
            category = parsed.get("category", "")
            confidence = float(parsed.get("confidence", 0.7))
            if category in self.class_names:
                return category, max(min(confidence, 0.99), 0.5)
        except (KeyError, ValueError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            logger.warning("LLM classification failed, falling back to heuristics: %s", exc)
        return None


class BugAnalysisAgent:
    def analyze(self, text: str, platform: str = "", rating: int = 0, app_version: str = "") -> Dict[str, str]:
        lowered = (text or "").lower()
        steps = []
        if "settings" in lowered:
            steps.append("Open Settings")
        if "login" in lowered or "sign in" in lowered:
            steps.append("Attempt sign-in")
        if "sync" in lowered:
            steps.append("Trigger data sync")
        if "update" in lowered:
            steps.append("Reproduce after recent update")
        if not steps:
            steps.append("Reproduce the issue from the reported workflow")

        severity = "High" if rating <= 2 else "Medium"
        technical_details = f"Platform: {platform or 'Unknown'}; App version: {app_version or 'Unknown'}; Steps: {' -> '.join(steps)}"
        return {"severity": severity, "technical_details": technical_details, "repro_steps": " -> ".join(steps)}


class FeatureExtractorAgent:
    def extract(self, text: str) -> Dict[str, object]:
        lowered = (text or "").lower()
        impact = "Medium"
        if any(term in lowered for term in ["dark mode", "offline", "notifications", "sync"]):
            impact = "High"
        demand = "High" if "would love" in lowered or "please add" in lowered else "Medium"
        return {"feature_name": "Feature request", "user_impact": impact, "demand_signal": demand}


class TicketCreatorAgent:
    def create_ticket(self, source_id: str, source_type: str, category: str, text: str, platform: str = "", app_version: str = "", rating: int = 0, confidence: float = 0.0, priority_hint: str = "") -> Dict[str, object]:
        base_priority = self._infer_priority(category, rating, priority_hint)
        title = self._build_title(category, text)
        return {
            "ticket_id": f"TKT-{source_id}",
            "source_id": source_id,
            "source_type": source_type,
            "category": category,
            "priority": base_priority,
            "title": title,
            "description": text,
            "platform": platform,
            "app_version": app_version,
            "rating": rating,
            "confidence": round(confidence, 2),
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }

    def _infer_priority(self, category: str, rating: int, priority_hint: str = "") -> str:
        if priority_hint:
            return priority_hint
        if category == "Bug" and rating <= 2:
            return "Critical"
        if category == "Bug":
            return "High"
        if category == "Feature Request":
            return "Medium"
        if category == "Complaint":
            return "Medium"
        return "Low"

    def _build_title(self, category: str, text: str) -> str:
        lowered = (text or "").lower()
        if category == "Bug":
            if "crash" in lowered:
                return "App crash issue reported"
            if "login" in lowered:
                return "Login issue reported"
            if "sync" in lowered:
                return "Data sync issue reported"
            return "Reported application bug"
        if category == "Feature Request":
            return "Feature request from user feedback"
        if category == "Praise":
            return "Positive feedback received"
        if category == "Complaint":
            return "User complaint logged"
        return "Feedback item reviewed"


class QualityCriticAgent:
    def review(self, ticket: Dict[str, object]) -> Dict[str, object]:
        issues = []
        if not ticket.get("title"):
            issues.append("missing_title")
        if not ticket.get("description"):
            issues.append("missing_description")
        if ticket.get("category") == "Bug" and not ticket.get("platform"):
            issues.append("missing_platform")
        approved = not issues
        return {"approved": approved, "issues": issues, "reviewed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}


class FeedbackProcessingPipeline:
    def __init__(self, use_llm: bool = False, api_key: Optional[str] = None, model: Optional[str] = None, api_base: Optional[str] = None):
        self.reader = CSVReaderAgent()
        self.classifier = FeedbackClassifierAgent(use_llm=use_llm, api_key=api_key, model=model, api_base=api_base)
        self.bug_agent = BugAnalysisAgent()
        self.feature_agent = FeatureExtractorAgent()
        self.ticket_creator = TicketCreatorAgent()
        self.critic = QualityCriticAgent()
        self.base_dir = Path(__file__).resolve().parent

    def process_all(self, data_dir: str = "data", output_dir: str = "output", overwrite: bool = False) -> Dict[str, str]:
        data_path = Path(data_dir)
        output_path = Path(output_dir)
        if not data_path.is_absolute():
            data_path = self.base_dir / data_path
        if not output_path.is_absolute():
            output_path = self.base_dir / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        reviews = self.reader.read_reviews(str(data_path / "app_store_reviews.csv"))
        emails = self.reader.read_support_emails(str(data_path / "support_emails.csv"))
        expected = pd.read_csv(str(data_path / "expected_classifications.csv"))

        tickets = []
        log_rows = []
        for item in reviews + emails:
            category, confidence = self.classifier.classify(item.text, platform=item.platform, rating=item.rating)
            analysis = {}
            if category == "Bug":
                analysis = self.bug_agent.analyze(item.text, platform=item.platform, rating=item.rating, app_version=item.app_version)
            elif category == "Feature Request":
                analysis = self.feature_agent.extract(item.text)

            ticket = self.ticket_creator.create_ticket(
                source_id=item.source_id,
                source_type=item.source_type,
                category=category,
                text=item.text,
                platform=item.platform or item.subject,
                app_version=item.app_version,
                rating=item.rating,
                confidence=confidence,
                priority_hint=item.priority,
            )
            review = self.critic.review(ticket)
            ticket["approval_status"] = "approved" if review["approved"] else "needs_review"
            ticket["review_notes"] = ";".join(review["issues"])
            ticket["llm_used"] = bool(getattr(self.classifier, "last_llm_used", False))
            if category == "Bug":
                ticket["technical_details"] = analysis.get("technical_details", "")
            elif category == "Feature Request":
                ticket["technical_details"] = json.dumps(analysis)
            tickets.append(ticket)
            log_rows.append({
                "source_id": item.source_id,
                "source_type": item.source_type,
                "category": category,
                "confidence": confidence,
                "approved": review["approved"],
                "review_issues": ";".join(review["issues"]),
                "llm_used": bool(getattr(self.classifier, "last_llm_used", False)),
            })

        tickets_df = pd.DataFrame(tickets)
        tickets_df.to_csv(output_path / "generated_tickets.csv", index=False)
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(output_path / "processing_log.csv", index=False)

        metrics = self._build_metrics(expected, tickets_df)
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(output_path / "metrics.csv", index=False)

        return {
            "generated_tickets.csv": str(output_path / "generated_tickets.csv"),
            "processing_log.csv": str(output_path / "processing_log.csv"),
            "metrics.csv": str(output_path / "metrics.csv"),
        }

    def _build_metrics(self, expected: pd.DataFrame, tickets_df: pd.DataFrame) -> List[Dict[str, object]]:
        expected_map = {
            (row["source_id"], row["source_type"]): row["category"]
            for _, row in expected.iterrows()
        }
        matches = 0
        total = 0
        for _, ticket in tickets_df.iterrows():
            total += 1
            expected_category = expected_map.get((ticket["source_id"], ticket["source_type"]))
            if expected_category and expected_category == ticket["category"]:
                matches += 1
        return [{"metric": "classification_accuracy", "value": round(matches / total, 3) if total else 0.0}]


if __name__ == "__main__":
    pipeline = FeedbackProcessingPipeline()
    outputs = pipeline.process_all(data_dir="data", output_dir="output", overwrite=True)
    print(json.dumps(outputs, indent=2))
