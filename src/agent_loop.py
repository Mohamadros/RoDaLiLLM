"""
Agentic Red-Teamer Feedback Loop.
Iteratively attacks the record linkage matcher until evasion is achieved.
"""

from typing import List, Dict, Any
from src.models import CustomerRecord
from src.llm_generator import LLMRecordRewriter
from src.perturbations import perturb_record_heuristically
from src.marini_client import MariniLinkageClient


class AgenticRedTeamer:
    def __init__(
        self,
        llm_provider: str = None,
        evasion_threshold: float = 0.75,
        max_iterations: int = 3,
    ):
        self.rewriter = LLMRecordRewriter(provider=llm_provider)
        self.marini_client = MariniLinkageClient()
        self.marini_client.authenticate()
        self.evasion_threshold = evasion_threshold
        self.max_iterations = max_iterations

    def attack_record(self, clean_record: CustomerRecord) -> Dict[str, Any]:
        """
        Executes the iterative feedback loop:
        1. Round 1: Pure LLM Persona Rewrite (Structural & Semantic shift)
        2. Round 2: LLM Rewrite + Optical / OCR Glitches
        3. Round 3: LLM Rewrite + OCR + Keyboard adjacent slips (Targeting prefixes)
        """
        history = []
        personas = ["rushed_mobile", "legacy_erp", "semantic_shift"]

        # Baseline evaluation (Clean vs Clean)
        baseline_eval = self.marini_client.run_linkage_evaluation(
            clean_record, clean_record, threshold=self.evasion_threshold
        )

        candidate = clean_record
        success = False

        for iteration in range(1, self.max_iterations + 1):
            chosen_persona = personas[(iteration - 1) % len(personas)]

            # Step 1: Semantic transformation via LLM
            candidate = self.rewriter.rewrite(clean_record, persona=chosen_persona)

            # Step 2: Stack physical noise if previous iteration did not evade
            if iteration == 2:
                candidate = perturb_record_heuristically(candidate, attack_type="ocr")
            elif iteration >= 3:
                candidate = perturb_record_heuristically(candidate, attack_type="hybrid")

            # Step 3: Test against Marini API / Matcher
            eval_result = self.marini_client.run_linkage_evaluation(
                clean_record, candidate, threshold=self.evasion_threshold
            )

            similarity = eval_result["similarity_score"]
            matched = eval_result["matched"]
            evaded = not matched

            step_summary = {
                "iteration": iteration,
                "strategy": f"LLM ({chosen_persona})" + (" + OCR" if iteration == 2 else " + Hybrid Noise" if iteration >= 3 else ""),
                "candidate_record": candidate.model_dump(),
                "similarity_score": similarity,
                "matched_by_engine": matched,
                "evasion_successful": evaded,
                "blocking_intact": eval_result.get("blocking_match", True),
            }
            history.append(step_summary)

            if evaded:
                success = True
                break

        return {
            "clean_record": clean_record.model_dump(),
            "final_adversarial_record": candidate.model_dump(),
            "baseline_score": baseline_eval["similarity_score"],
            "final_score": history[-1]["similarity_score"],
            "score_drop": round(baseline_eval["similarity_score"] - history[-1]["similarity_score"], 4),
            "evasion_successful": success,
            "total_rounds": len(history),
            "history": history,
        }
