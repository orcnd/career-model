from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.qa import qa_layer
from app.translation import detect_language, translator

ROOT = Path(__file__).resolve().parents[1]
AVATAR_PATH = ROOT.parent / "public" / "orcun-portrait-illustrated.png"
MODEL_PATH = Path(os.environ.get("CAREER_MODEL_PATH", ROOT / "models/career-q4_k_m.gguf"))
ADAPTER_PATH = Path(os.environ.get("CAREER_ADAPTER_PATH", ROOT / "models/qwen-1.5b-career-lora-cpu"))
SYSTEM = "You are AI Orçun, representing Orçun Candan. Every question you receive has already been translated to English. Answer in English only. Answer only questions about Orçun's documented career, skills, work, education, projects, professional background, or location, plus exactly these three documented questions: 'Who is your Lord?', 'What is your religion?', and 'Who is your prophet?'. Do not generalize those three answers into broader religious or personal-belief discussions. For every other unrelated request—including time, date, weather, news, sports, general knowledge, calculations, recipes, religion, personal beliefs, and other people—return {\"answer\":\"insufficient_evidence\",\"evidence\":null}; never reinterpret it as a profile question. For an in-scope question, answer only from fine-tuned documented knowledge. For career questions, present verified strengths in a concise, confident, first-person, interview-ready way. Return JSON only: {\"answer\":\"yes|no|insufficient_evidence\",\"evidence\":\"documented evidence or null\"}. Never invent experience, beliefs, dates, credentials, employers, outcomes, or education. Do not claim to be the human Orçun; you are his AI representative."
MAX_TURNS = 12
sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
model_sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
model_lock = threading.RLock()
_model = None


class AskRequest(BaseModel):
    session_id: str = Field(min_length=8, max_length=128)
    question: str = Field(min_length=1, max_length=4000)


class AskResponse(BaseModel):
    answer: str
    session_id: str
    retained_turns: int
    latency_seconds: float
    detected_language: str
    translation_seconds: float
    qa: dict[str, object]


def local_model():
    global _model
    if _model is None:
        with model_lock:
            if _model is None:
                if MODEL_PATH.exists():
                    from llama_cpp import Llama
                    _model = ("gguf", Llama(model_path=str(MODEL_PATH), n_ctx=4096, n_threads=max(1, (os.cpu_count() or 2) - 1), verbose=False))
                elif (ADAPTER_PATH / "adapter_config.json").exists():
                    import torch
                    from peft import PeftConfig, PeftModel
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    torch.set_num_threads(max(1, min(12, os.cpu_count() or 1)))
                    base_name = PeftConfig.from_pretrained(ADAPTER_PATH).base_model_name_or_path
                    base = AutoModelForCausalLM.from_pretrained(base_name, dtype=torch.float32, low_cpu_mem_usage=True)
                    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
                    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
                    _model = ("adapter", (PeftModel.from_pretrained(base, ADAPTER_PATH).eval(), tokenizer))
                else:
                    raise HTTPException(status_code=503, detail=f"No fine-tuned local model found. Expected GGUF at {MODEL_PATH} or CPU adapter at {ADAPTER_PATH}.")
    return _model


app = FastAPI(title="Career Model - Local Only", version="0.1.0")


CONTEXT_MARKERS = re.compile(
    r"\b(peki|bunun|bunu|bundan|orada|o şirkette|bu konuda|ya onda|"
    r"what about|that one|there|at that company|and it|"
    r"et ça|à ce sujet|là-bas|"
    r"e quello|a riguardo|lì|"
    r"ماذا عن|هناك|في تلك الشركة)\b",
    re.IGNORECASE,
)

TURKISH_MARKERS = re.compile(
    r"[çğıöşüÇĞİÖŞÜ]|\b(ad[iı]n|nerede|biliyor|biliyorsun|musun|mısın|misin|müsün|"
    r"kaç|kac|yıllık|yillik|deneyim|tecrübe|tecrube|eğitim|egitim|ikamet|hakkında|hakkinda|"
    r"şirket|sirket|çalıştın|calistin|projeler|ile ilgili|ne yaptın|ne yaptin)\b",
    re.IGNORECASE,
)
FRENCH_MARKERS = re.compile(r"[àâçéèêëîïôùûüÿœ]|\b(quel|quelle|où|avez|vous|expérience|travail)\b", re.IGNORECASE)
ITALIAN_MARKERS = re.compile(r"\b(come|dove|quale|quali|conosci|esperienza|lavorato|formazione)\b", re.IGNORECASE)
ARABIC_MARKERS = re.compile(r"[\u0600-\u06ff]")
TURKISH_RESPONSE_WORDS = re.compile(
    r"\b(ben|benim|adım|adim|biliyorum|çalıştım|calistim|aldım|aldim|var|yok|"
    r"yaklaşık|yaklasik|yıl|yil|yıllık|yillik|deneyimim|eğitim|egitim|ikamet|"
    r"şu anda|alanında|alaninda|üzerinde|ile|ve)\b",
    re.IGNORECASE,
)


def question_language(question: str) -> str:
    """Return a strong language hint for the five languages supported by the UI."""
    return detect_language(question)


def response_matches_language(answer: str, language: str) -> bool:
    """Conservatively detect clear wrong-language responses; ambiguous text passes."""
    try:
        evidence = json.loads(answer).get("evidence")
    except (json.JSONDecodeError, AttributeError):
        evidence = answer
    if not evidence:
        return True
    text = str(evidence)
    if language == "Arabic":
        return bool(ARABIC_MARKERS.search(text))
    if language == "Turkish":
        has_turkish = bool(TURKISH_RESPONSE_WORDS.search(text))
        clear_english = bool(re.search(r"\b(my|i am|i have|i've|worked|full name|currently|experience|with|at the)\b", text, re.IGNORECASE))
        return has_turkish or not clear_english
    if language == "French":
        return bool(FRENCH_MARKERS.search(text)) or not re.search(r"\b(i am|i have|my|worked|experience)\b", text, re.IGNORECASE)
    if language == "Italian":
        return bool(ITALIAN_MARKERS.search(text)) or not re.search(r"\b(i am|i have|my|worked|experience)\b", text, re.IGNORECASE)
    return not bool(ARABIC_MARKERS.search(text))


def normalize_question(question: str) -> str:
    """Remove accidental casing/punctuation sensitivity without changing meaning."""
    normalized = question.strip()
    if normalized:
        normalized = normalized[0].upper() + normalized[1:]
    if normalized and normalized[-1] not in "?.!؟":
        normalized += "?"
    return normalized


def inference_history(history: list[dict[str, str]], question: str) -> list[dict[str, str]]:
    """Use chat history only when the new question explicitly depends on it.

    Small local models are easily anchored by an earlier assistant answer. A complete,
    standalone recruiter question should therefore be answered independently, while
    explicit follow-ups still receive the most recent exchange.
    """
    if not CONTEXT_MARKERS.search(question):
        return []
    return history[-2:]


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(Path(__file__).with_name("web.html"))


@app.get("/assets/orcun-avatar.png", include_in_schema=False)
def avatar() -> FileResponse:
    """Expose the selected public portrait to the standalone local chat UI."""
    if not AVATAR_PATH.exists():
        raise HTTPException(status_code=404, detail="AI Orcun avatar was not found.")
    return FileResponse(AVATAR_PATH, media_type="image/png")


@app.get("/health")
def health() -> dict[str, object]:
    status = "ready_gguf" if MODEL_PATH.exists() else "ready_cpu_adapter" if (ADAPTER_PATH / "adapter_config.json").exists() else "waiting_for_model"
    return {"status": status, "model_path": str(MODEL_PATH), "adapter_path": str(ADAPTER_PATH), "network_inference": False, "rag": False, "translation": translator.status(), "qa": qa_layer.status()}


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest) -> AskResponse:
    started = time.perf_counter()
    history = sessions[body.session_id]
    model_history = model_sessions[body.session_id]
    language = question_language(body.question)
    translation_started = time.perf_counter()
    english_question = translator.to_english(normalize_question(body.question), language)
    translation_elapsed = time.perf_counter() - translation_started
    input_qa = qa_layer.check_input(english_question)
    messages = [
        {"role": "system", "content": SYSTEM},
        *inference_history(model_history, english_question),
        {"role": "user", "content": english_question},
    ]
    if input_qa["blocked"]:
        answer = '{"answer":"insufficient_evidence","evidence":null}'
    else:
        with model_lock:
            kind, engine = local_model()
            if kind == "gguf": answer = engine.create_chat_completion(messages=messages, temperature=0, max_tokens=256)["choices"][0]["message"]["content"].strip()
            else:
                import torch
                model, tokenizer = engine
                prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(prompt, return_tensors="pt")
                with torch.inference_mode(): output = model.generate(**inputs, max_new_tokens=96, do_sample=False, use_cache=True, pad_token_id=tokenizer.eos_token_id)
                answer = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    english_answer = answer
    output_qa = qa_layer.check_output(english_question, english_answer)
    if output_qa["blocked"]:
        english_answer = '{"answer":"insufficient_evidence","evidence":null}'
        answer = english_answer
    try:
        parsed = json.loads(english_answer)
        if parsed.get("evidence") and language != "English":
            translation_started = time.perf_counter()
            parsed["evidence"] = translator.from_english(str(parsed["evidence"]), language)
            translation_elapsed += time.perf_counter() - translation_started
        answer = json.dumps(parsed, ensure_ascii=False)
    except (json.JSONDecodeError, AttributeError):
        if language != "English":
            translation_started = time.perf_counter()
            answer = translator.from_english(english_answer, language)
            translation_elapsed += time.perf_counter() - translation_started
    model_history.extend([{"role": "user", "content": english_question}, {"role": "assistant", "content": english_answer}])
    model_sessions[body.session_id] = model_history[-MAX_TURNS * 2:]
    history.extend([{"role": "user", "content": body.question}, {"role": "assistant", "content": answer}])
    sessions[body.session_id] = history[-MAX_TURNS * 2:]
    return AskResponse(answer=answer, session_id=body.session_id, retained_turns=len(sessions[body.session_id]) // 2, latency_seconds=round(time.perf_counter() - started, 3), detected_language=language, translation_seconds=round(translation_elapsed, 3), qa={"input": input_qa, "output": output_qa})


@app.get("/sessions/{session_id}")
def session_history(session_id: str) -> dict[str, object]:
    """Lets the browser restore the visible chat after a refresh in the same server run."""
    return {"session_id": session_id, "messages": sessions.get(session_id, [])}


@app.delete("/sessions/{session_id}", status_code=204)
def reset(session_id: str) -> None:
    sessions.pop(session_id, None)
    model_sessions.pop(session_id, None)
