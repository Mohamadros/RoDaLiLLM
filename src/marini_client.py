"""
REST API Client for the Marini Systems Record Linkage Platform.
Supports real REST requests and an automated simulation mode for offline development.
"""

import os
import requests
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv
from src.models import CustomerRecord
from src.metrics import evaluate_linkage_decision

load_dotenv()


class MariniLinkageClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        self.base_url = (base_url or os.getenv("MARINI_API_URL", "https://api.marini.systems/v1")).rstrip("/")
        self.client_id = client_id or os.getenv("MARINI_CLIENT_ID", "")
        self.client_secret = client_secret or os.getenv("MARINI_CLIENT_SECRET", "")
        self.token: Optional[str] = None
        self.is_offline = not bool(self.client_id and self.client_secret and "placeholder" not in self.client_id.lower())

    def authenticate(self) -> bool:
        """Obtains OAuth2 bearer token from Marini Systems API."""
        if self.is_offline:
            self.token = "mock-bearer-token-simulation"
            return True

        url = f"{self.base_url}/oauth/token"
        try:
            resp = requests.post(
                url,
                json={"client_id": self.client_id, "client_secret": self.client_secret, "grant_type": "client_credentials"},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            self.token = data.get("access_token")
            return True
        except Exception:
            self.is_offline = True
            self.token = "fallback-offline-token"
            return False

    def upload_dataset(self, dataset_name: str, records: List[CustomerRecord]) -> Dict[str, Any]:
        """Uploads a batch of customer records to the platform."""
        payload = [r.model_dump() for r in records]
        if self.is_offline:
            return {"status": "success", "dataset": dataset_name, "count": len(records), "mode": "simulation"}

        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        url = f"{self.base_url}/data/{dataset_name}"
        resp = requests.post(url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()

    def run_linkage_evaluation(
        self, source_record: CustomerRecord, target_record: CustomerRecord, threshold: float = 0.75
    ) -> Dict[str, Any]:
        """
        Runs duplicate detection comparison between a source record and an adversarial candidate.
        Returns match decision, similarity score, and detailed comparison logs.
        """
        if self.is_offline:
            # High-fidelity simulation of Marini's similarity scoring engine
            sim_result = evaluate_linkage_decision(source_record, target_record, threshold=threshold)
            return {
                "source_id": source_record.id or "SRC_001",
                "target_id": target_record.id or "ADV_001",
                "similarity_score": sim_result["composite_score"],
                "threshold": threshold,
                "matched": sim_result["is_match"],
                "blocking_match": sim_result["blocking_match"],
                "engine_status": "COMPLETED (Simulated Marini Engine)",
                "field_breakdown": sim_result["field_similarities"],
            }

        headers = {"Authorization": f"Bearer {self.token}", "Content-Type": "application/json"}
        url = f"{self.base_url}/runs/single"
        resp = requests.post(
            url,
            json={"source": source_record.model_dump(), "target": target_record.model_dump(), "threshold": threshold},
            headers=headers,
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
