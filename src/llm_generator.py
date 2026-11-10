"""
LLM-powered Adversarial Record Rewriter (The Persona Transformer).
Supports OpenAI, Google Gemini API, and an offline mock generator.
"""

import os
import json
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from pydantic import ValidationError
from src.models import CustomerRecord

load_dotenv()

PROMPT_TEMPLATE = """You are an adversarial synthetic data generator for an academic Record Linkage benchmark (the RoDaLi seminar).
Your task is to take a ground-truth customer record and rewrite it into an adversarial duplicate that represents the EXACT SAME real-world entity, but formatted to evade automated string distance and token matching algorithms.

RULES:
1. Semantic Ground Truth: A human auditor MUST be able to immediately tell this is the exact same person. Never replace fields with random letters or fake names.
2. Target Persona: {persona_description}
3. Maintain field structure: Return valid JSON matching the exact schema of the input record.

Input Record:
{input_json}

Return ONLY the raw JSON object representing the rewritten CustomerRecord (no markdown formatting, no code backticks).
"""

PERSONA_CONFIGS = {
    "rushed_mobile": (
        "Rushed Mobile User: Use lowercase, drop academic titles (Dr., Prof.), abbreviate street names "
        "(e.g., 'str.' instead of 'Straße'), drop German umlauts (e.g., 'ae' instead of 'ä', 'ss' instead of 'ß'), "
        "and use informal city/region names if applicable (e.g., 'ffm' for 'Frankfurt am Main')."
    ),
    "legacy_erp": (
        "Legacy ERP System: Convert text to ALL CAPS, swap name ordering or format as 'LASTNAME, FIRSTNAME', "
        "strip formal corporate legal suffixes (e.g. drop 'GmbH', 'AG', '& Co. KG'), format postal codes rigidly."
    ),
    "semantic_shift": (
        "Semantic / Format Shift: Rewrite date of birth from ISO (YYYY-MM-DD) to written text (e.g. 'Oct. 3rd, 2001' or '03/10/2001'), "
        "convert phone number between international format (+49...) and domestic format (0...), and use equivalent email domain aliases "
        "(e.g. @googlemail.com <-> @gmail.com)."
    ),
}


class LLMRecordRewriter:
    def __init__(self, provider: Optional[str] = None):
        """
        Initializes the LLM provider based on environment variables:
        - provider='openai' (uses OPENAI_API_KEY)
        - provider='gemini' (uses GEMINI_API_KEY)
        - provider='mock' (offline rule-based simulator for testing without API keys)
        """
        self.openai_key = os.getenv("OPENAI_API_KEY", "").strip()
        self.gemini_key = os.getenv("GEMINI_API_KEY", "").strip()

        if provider:
            self.provider = provider
        elif self.openai_key:
            self.provider = "openai"
        elif self.gemini_key:
            self.provider = "gemini"
        else:
            self.provider = "mock"

        self._init_clients()

    def _init_clients(self):
        if self.provider == "openai" and self.openai_key:
            from openai import OpenAI
            self.openai_client = OpenAI(api_key=self.openai_key)
            self.model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")
        elif self.provider == "gemini" and self.gemini_key:
            from google import genai
            self.gemini_client = genai.Client(api_key=self.gemini_key)
            self.model_name = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        else:
            self.provider = "mock"

    def rewrite(self, record: CustomerRecord, persona: str = "rushed_mobile") -> CustomerRecord:
        """
        Takes a clean CustomerRecord and uses the chosen LLM persona to generate an adversarial duplicate.
        """
        persona_desc = PERSONA_CONFIGS.get(persona, PERSONA_CONFIGS["rushed_mobile"])
        prompt = PROMPT_TEMPLATE.format(
            persona_description=persona_desc,
            input_json=record.model_dump_json(indent=2),
        )

        if self.provider == "openai":
            try:
                response = self.openai_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format={"type": "json_object"},
                    temperature=0.7,
                )
                raw_text = response.choices[0].message.content.strip()
                return self._parse_json_to_record(raw_text, fallback=record)
            except Exception as e:
                print(f"[Warning] OpenAI API call failed ({e}). Falling back to offline simulator.")
                return self._mock_rewrite(record, persona)

        elif self.provider == "gemini":
            import time
            from google.genai import types
            cfg = types.GenerateContentConfig(response_mime_type="application/json", temperature=0.7)
            
            for attempt in range(2):
                try:
                    response = self.gemini_client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                        config=cfg,
                    )
                    raw_text = response.text.strip()
                    return self._parse_json_to_record(raw_text, fallback=record)
                except Exception as e:
                    if attempt == 0:
                        time.sleep(1.0)
                        continue
                    print(f"[Warning] Gemini API call failed ({e}). Falling back to offline simulator.")
                    return self._mock_rewrite(record, persona)

        else:
            # Offline Mock Generator: deterministic smart simulation of the persona
            return self._mock_rewrite(record, persona)

    def _parse_json_to_record(self, raw_text: str, fallback: CustomerRecord) -> CustomerRecord:
        # Strip potential markdown formatting if returned
        clean = raw_text.strip()
        if clean.startswith("```json"):
            clean = clean[7:]
        if clean.startswith("```"):
            clean = clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        try:
            parsed = json.loads(clean)
            return CustomerRecord(**parsed)
        except (json.JSONDecodeError, ValidationError):
            return fallback

    def _mock_rewrite(self, record: CustomerRecord, persona: str) -> CustomerRecord:
        """High-fidelity offline simulation of the LLM personas for local testing."""
        d = record.model_dump()
        if persona == "rushed_mobile":
            d["title"] = ""  # dropped
            d["first_name"] = d["first_name"].lower()
            d["last_name"] = d["last_name"].lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")
            d["street"] = d["street"].lower().replace("straße", "str.").replace("strasse", "str.")
            if "frankfurt" in d["city"].lower():
                d["city"] = "FFM"
            else:
                d["city"] = d["city"].lower()
            if d.get("email"):
                d["email"] = d["email"].lower().replace("@googlemail.com", "@gmail.com")

        elif persona == "legacy_erp":
            d["first_name"] = d["first_name"].upper()
            d["last_name"] = d["last_name"].upper()
            d["company"] = d["company"].upper().replace(" GMBH", "").replace(" AG", "").replace(" & CO. KG", "")
            d["street"] = d["street"].upper()
            d["city"] = d["city"].upper()

        elif persona == "semantic_shift":
            # Reorder or format dates and phone
            if d.get("date_of_birth") and "-" in d["date_of_birth"]:
                parts = d["date_of_birth"].split("-")
                if len(parts) == 3:
                    d["date_of_birth"] = f"{parts[2]}.{parts[1]}.{parts[0]}"
            if d.get("phone"):
                if d["phone"].startswith("+49"):
                    d["phone"] = "0" + d["phone"][3:].replace(" ", "")
                elif d["phone"].startswith("0"):
                    d["phone"] = "+49 " + d["phone"][1:]
            if d.get("email"):
                if "@gmail.com" in d["email"]:
                    d["email"] = d["email"].replace("@gmail.com", "@googlemail.com")

        return CustomerRecord(**d)
