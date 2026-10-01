import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from career_model.core import INSUFFICIENT, redact
from app.server import inference_history, normalize_question, question_language, response_matches_language
from scripts.dataset.generate import role_examples, split, tech_examples


def profile():
    return {"subject": "Avery", "technologies": [{"name": "Laravel", "evidence": "Documented Laravel migration work."}], "roles": [{"title": "Engineering Lead", "evidence": "Led a documented team."}], "negative_technologies": ["COBOL"]}


def test_redaction_removes_contact_data():
    clean = redact("Reach me at name@example.com or +90 555 000 00 00")
    assert "example.com" not in clean and "555" not in clean


def test_unsupported_examples_refuse_claims():
    examples = tech_examples(profile())
    assert any(x.label == "unsupported" and x.answer == INSUFFICIENT for x in examples)


def test_fact_families_never_cross_splits():
    groups = split(tech_examples(profile()) + role_examples(profile()))
    locations = {x.family: name for name, values in groups.items() for x in values}
    assert len(locations) == len({x.family for values in groups.values() for x in values})


def test_standalone_question_is_not_anchored_by_previous_answer():
    history = [
        {"role": "user", "content": "TypeScript biliyor musun?"},
        {"role": "assistant", "content": "TypeScript ve React ile çalıştım."},
    ]
    assert inference_history(history, "AI ile ilgili ne biliyorsun?") == []


def test_explicit_follow_up_keeps_latest_exchange():
    history = [
        {"role": "user", "content": "PHP biliyor musun?"},
        {"role": "assistant", "content": "Evet, biliyorum."},
    ]
    assert inference_history(history, "Peki bununla hangi projelerde çalıştın?") == history


def test_turkish_is_detected_without_diacritics():
    assert question_language("adin ne") == "Turkish"
    assert question_language("php biliyor musun") == "Turkish"
    assert question_language("nerede egitim aldin") == "Turkish"


def test_clear_english_answer_fails_turkish_language_check():
    answer = '{"answer":"yes","evidence":"My full name is Orcun Candan."}'
    assert not response_matches_language(answer, "Turkish")


def test_question_normalization_is_stable():
    assert normalize_question("adin ne") == "Adin ne?"
    assert normalize_question("Adin ne?") == "Adin ne?"
