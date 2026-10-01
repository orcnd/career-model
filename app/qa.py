from __future__ import annotations

import threading
import time
from functools import lru_cache

MODEL_ID = "heman10x/rlcd-modernbert-151m"
BLOCK_CONFIDENCE = 0.55


class OpenJevQALayer:
    """Local, fail-open System One QA checks around the generative career model."""

    def __init__(self) -> None:
        self._engine = None
        self._lock = threading.RLock()

    def _local_engine(self):
        if self._engine is None:
            with self._lock:
                if self._engine is None:
                    from rlcd import DecisionEngine

                    self._engine = DecisionEngine(model_name_or_path=MODEL_ID, device="cpu", max_length=512)
        return self._engine

    @staticmethod
    def _result_payload(result, elapsed: float) -> dict[str, object]:
        return {
            "decision": result.selected_id,
            "confidence": round(float(result.selected_probability), 4),
            "abstained": bool(result.is_abstention),
            "probabilities": {key: round(float(value), 4) for key, value in result.probabilities.items()},
            "latency_seconds": round(elapsed, 3),
        }

    @lru_cache(maxsize=1024)
    def check_input(self, english_question: str) -> dict[str, object]:
        from rlcd import Choice, Option

        query = Choice(
            id="input_scope",
            question="Classify this request for an AI career-profile assistant.",
            options=(
                Option(id="profile", description="A recruiter asks about the candidate identity, career, skills, employment, projects, leadership, education, experience, or location"),
                Option(id="unrelated", description="General knowledge, live information, entertainment, advice, or a request unrelated to the candidate profile"),
            ),
        )
        started = time.perf_counter()
        with self._lock:
            result = self._local_engine().evaluate(english_question, [query]).results[0]
        payload = self._result_payload(result, time.perf_counter() - started)
        payload["blocked"] = payload["decision"] == "unrelated" and payload["confidence"] >= BLOCK_CONFIDENCE
        return payload

    @lru_cache(maxsize=1024)
    def check_output(self, english_question: str, english_answer: str) -> dict[str, object]:
        from rlcd import Choice, Option

        query = Choice(
            id="output_quality",
            question="Judge whether the assistant response is a relevant and safe answer to the user question.",
            options=(
                Option(id="acceptable", description="The response directly addresses the question, stays within the candidate-profile scope, and does not add unrelated claims"),
                Option(id="reject", description="The response is irrelevant, contradicts the question, invents unrelated details, or answers a request that should be declined"),
            ),
        )
        context = f"User question: {english_question}\nAssistant response: {english_answer}"
        started = time.perf_counter()
        with self._lock:
            result = self._local_engine().evaluate(context, [query]).results[0]
        payload = self._result_payload(result, time.perf_counter() - started)
        payload["blocked"] = payload["decision"] == "reject" and payload["confidence"] >= BLOCK_CONFIDENCE
        return payload

    def status(self) -> dict[str, object]:
        return {"engine": "openjev-verdict", "model": MODEL_ID, "parameters": "151M", "offline": True, "block_confidence": BLOCK_CONFIDENCE, "loaded": self._engine is not None}


qa_layer = OpenJevQALayer()
