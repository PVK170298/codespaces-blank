import json
import logging
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import load_dotenv


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


class FeedbackClassifierAgent:
    def __init__(self, use_llm: bool = True, api_key: Optional[str] = None, model: Optional[str] = None, api_base: Optional[str] = None):
        self.class_names = ["Bug", "Feature Request", "Praise", "Complaint", "Spam"]
        self.use_llm = use_llm
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        self.api_base = (api_base or os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1/chat/completions")).rstrip("/")
        self.last_llm_used = False
        self.last_error = ""

    def classify(self, text: str, platform: str = "", rating: int = 0) -> Tuple[str, float]:
        self.last_llm_used = False
        if self.use_llm and self.api_key:
            llm_result = self._classify_with_llm(text, platform, rating)
            if llm_result:
                self.last_llm_used = True
                return llm_result
            logger.warning("Groq request failed or returned invalid data; falling back to heuristic classification.")

        return self._classify_without_llm(text), 0.5

    def _classify_without_llm(self, text: str) -> str:
        lowered = (text or "").lower()
        if any(term in lowered for term in ["crash", "fail", "error", "login", "bug", "not working", "sync"]):
            return "Bug"
        if any(term in lowered for term in ["feature", "add", "dark mode", "please", "would love", "suggestion"]):
            return "Feature Request"
        if any(term in lowered for term in ["love", "great", "amazing", "awesome", "fantastic"]):
            return "Praise"
        if any(term in lowered for term in ["bad", "poor", "frustrating", "slow", "terrible", "expensive"]):
            return "Complaint"
        return "Spam"

    def _classify_with_llm(self, text: str, platform: str = "", rating: int = 0) -> Optional[Tuple[str, float]]:
        prompt = (
            "You are a support ticket triage assistant. Classify the feedback into exactly one of these categories: "
            "Bug, Feature Request, Praise, Complaint, Spam. Return JSON with keys 'category' and 'confidence'. "
            f"Feedback: {text} | platform: {platform or 'unknown'} | rating: {rating}"
        )
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You respond with compact JSON only."},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        try:
            req = urllib.request.Request(
                self.api_base,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.load(response)
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            parsed = json.loads(content)
            category = parsed.get("category", "")
            confidence = float(parsed.get("confidence", 0.7))
            if category in self.class_names:
                return category, max(min(confidence, 0.99), 0.5)
        except (KeyError, ValueError, TypeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
            self.last_error = f"Groq API error: {exc}"
            logger.warning("Groq classification failed: %s", exc)
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
    def __init__(self, use_llm: bool = True, api_key: Optional[str] = None, model: Optional[str] = None, api_base: Optional[str] = None):
        self.classifier = FeedbackClassifierAgent(use_llm=use_llm, api_key=api_key, model=model, api_base=api_base)
        self.bug_agent = BugAnalysisAgent()
        self.feature_agent = FeatureExtractorAgent()
        self.ticket_creator = TicketCreatorAgent()
        self.critic = QualityCriticAgent()
        self.base_dir = Path(__file__).resolve().parent

    def process_feedback_items(self, feedback_items: List[Dict[str, Any]], output_dir: str = "output") -> Dict[str, str]:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = self.base_dir / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        tickets = []
        log_rows = []
        for item in feedback_items:
            record = FeedbackRecord(
                source_id=str(item.get("source_id", "feedback")),
                source_type=str(item.get("source_type", "feedback")),
                text=str(item.get("text", "")),
                platform=str(item.get("platform", "")),
                rating=int(item.get("rating", 0)),
                app_version=str(item.get("app_version", "")),
                subject=str(item.get("subject", "")),
                sender_email=str(item.get("sender_email", "")),
                timestamp=str(item.get("timestamp", "")),
                priority=str(item.get("priority", "")),
            )
            category, confidence = self.classifier.classify(record.text, platform=record.platform, rating=record.rating)
            analysis = {}
            if category == "Bug":
                analysis = self.bug_agent.analyze(record.text, platform=record.platform, rating=record.rating, app_version=record.app_version)
            elif category == "Feature Request":
                analysis = self.feature_agent.extract(record.text)

            ticket = self.ticket_creator.create_ticket(
                source_id=record.source_id,
                source_type=record.source_type,
                category=category,
                text=record.text,
                platform=record.platform or record.subject,
                app_version=record.app_version,
                rating=record.rating,
                confidence=confidence,
                priority_hint=record.priority,
            )
            review = self.critic.review(ticket)
            ticket["approval_status"] = "approved" if review["approved"] else "needs_review"
            ticket["review_notes"] = ";".join(review["issues"])
            ticket["llm_used"] = bool(getattr(self.classifier, "last_llm_used", False))
            ticket["llm_error"] = getattr(self.classifier, "last_error", "")
            if category == "Bug":
                ticket["technical_details"] = analysis.get("technical_details", "")
            elif category == "Feature Request":
                ticket["technical_details"] = json.dumps(analysis)
            tickets.append(ticket)
            log_rows.append({
                "source_id": record.source_id,
                "source_type": record.source_type,
                "category": category,
                "confidence": confidence,
                "approved": review["approved"],
                "review_issues": ";".join(review["issues"]),
                "llm_used": bool(getattr(self.classifier, "last_llm_used", False)),
                "llm_error": getattr(self.classifier, "last_error", ""),
            })

        tickets_df = pd.DataFrame(tickets)
        tickets_df.to_csv(output_path / "generated_tickets.csv", index=False)
        log_df = pd.DataFrame(log_rows)
        log_df.to_csv(output_path / "processing_log.csv", index=False)

        metrics = [
            {"metric": "total_feedback_items", "value": len(feedback_items)},
            {"metric": "tickets_generated", "value": len(tickets_df)},
            {"metric": "approved_tickets", "value": int((tickets_df["approval_status"] == "approved").sum()) if not tickets_df.empty else 0},
            {"metric": "llm_classifications", "value": int(tickets_df["llm_used"].sum()) if not tickets_df.empty else 0},
        ]
        metrics_df = pd.DataFrame(metrics)
        metrics_df.to_csv(output_path / "metrics.csv", index=False)

        return {
            "generated_tickets.csv": str(output_path / "generated_tickets.csv"),
            "processing_log.csv": str(output_path / "processing_log.csv"),
            "metrics.csv": str(output_path / "metrics.csv"),
        }

    def process_all(self, feedback_items: Optional[List[Dict[str, Any]]] = None, data_dir: Optional[str] = None, output_dir: str = "output", overwrite: bool = False) -> Dict[str, str]:
        if feedback_items is not None:
            return self.process_feedback_items(feedback_items, output_dir=output_dir)

        if data_dir is None:
            data_dir = "data"

        data_path = Path(data_dir)
        if not data_path.is_absolute():
            data_path = self.base_dir / data_path

        reviews = self._read_reviews(str(data_path / "app_store_reviews.csv"))
        emails = self._read_support_emails(str(data_path / "support_emails.csv"))
        expected = pd.read_csv(str(data_path / "expected_classifications.csv"))

        items = [
            {
                "source_id": item.source_id,
                "source_type": item.source_type,
                "text": item.text,
                "platform": item.platform,
                "rating": item.rating,
                "app_version": item.app_version,
                "subject": item.subject,
                "sender_email": item.sender_email,
                "timestamp": item.timestamp,
                "priority": item.priority,
            }
            for item in reviews + emails
        ]
        outputs = self.process_feedback_items(items, output_dir=output_dir)
        self._write_metrics(expected, output_dir=output_dir)
        return outputs

    def _read_reviews(self, file_path: str) -> List[FeedbackRecord]:
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

    def _read_support_emails(self, file_path: str) -> List[FeedbackRecord]:
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

    def _write_metrics(self, expected: pd.DataFrame, output_dir: str) -> None:
        output_path = Path(output_dir)
        if not output_path.is_absolute():
            output_path = self.base_dir / output_path
        output_path.mkdir(parents=True, exist_ok=True)

        metrics_path = output_path / "metrics.csv"
        if not metrics_path.exists():
            metrics_path.write_text("metric,value\n", encoding="utf-8")

        current = pd.read_csv(metrics_path)
        expected_map = {(row["source_id"], row["source_type"]): row["category"] for _, row in expected.iterrows()}
        tickets = pd.read_csv(output_path / "generated_tickets.csv")
        matches = 0
        total = 0
        for _, ticket in tickets.iterrows():
            total += 1
            expected_category = expected_map.get((ticket["source_id"], ticket["source_type"]))
            if expected_category and expected_category == ticket["category"]:
                matches += 1

        if current.empty or "classification_accuracy" not in current["metric"].values:
            current = pd.concat([
                current,
                pd.DataFrame([{"metric": "classification_accuracy", "value": round(matches / total, 3) if total else 0.0}]),
            ], ignore_index=True)
        else:
            current.loc[current["metric"] == "classification_accuracy", "value"] = round(matches / total, 3) if total else 0.0
        current.to_csv(metrics_path, index=False)


if __name__ == "__main__":
    sample_feedback = [
        {"source_id": "demo_1", "source_type": "app_store_review", "text": "The app crashes when I open settings", "platform": "iOS", "rating": 1},
        {"source_id": "demo_2", "source_type": "support_email", "text": "Please add dark mode support", "platform": "Email", "rating": 4},
    ]
    pipeline = FeedbackProcessingPipeline(use_llm=True)
    outputs = pipeline.process_feedback_items(sample_feedback, output_dir="output")
    print(json.dumps(outputs, indent=2))
