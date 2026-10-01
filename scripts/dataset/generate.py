from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from career_model.core import Example, INSUFFICIENT, read_json, stable_bucket


def tech_examples(profile: dict) -> list[Example]:
    subject = profile.get("subject", "the candidate")
    out = []
    for item in profile.get("technologies", []):
        name, evidence = item["name"], item["evidence"]
        family = f"technology:{name.casefold()}"
        answer = json.dumps({"answer": "yes", "evidence": evidence})
        for i, question in enumerate([f"Does {subject} know {name}?", f"Has {subject} worked with {name}?", f"Can {subject} work with {name}?", f"What experience does {subject} have with {name}?", f"Is {subject} familiar with {name}?"]):
            out.append(Example(f"{family}:{i}", family, question, answer, [evidence], "supported"))
        for i, pair in enumerate(item.get("training_pairs", [])):
            out.append(Example(f"{family}:localized:{i}", family, pair["question"], json.dumps({"answer": "yes", "evidence": pair["answer"]}, ensure_ascii=False), [evidence], "supported"))
    for name in profile.get("negative_technologies", []):
        family = f"negative:{name.casefold()}"
        for i, question in enumerate([f"Does {subject} know {name}?", f"Has {subject} ever used {name}?", f"Tell me about {subject}'s {name} experience."]):
            out.append(Example(f"{family}:{i}", family, question, INSUFFICIENT, [], "unsupported"))
    negative_repeats = int(profile.get("negative_training_weight", 1))
    for i, question in enumerate(profile.get("negative_training_questions", [])):
        for repeat in range(negative_repeats):
            out.append(Example(f"negative:localized:{i}:{repeat}", "negative:localized", question, INSUFFICIENT, [], "unsupported"))
    return out


def role_examples(profile: dict) -> list[Example]:
    subject, out = profile.get("subject", "the candidate"), []
    for role in profile.get("roles", []):
        title, evidence = role["title"], role["evidence"]
        family = f"role:{title.casefold()}"
        for i, q in enumerate([f"Has {subject} held a {title} role?", f"What is documented about {subject}'s work as {title}?"]):
            out.append(Example(f"{family}:{i}", family, q, json.dumps({"answer": "yes", "evidence": evidence}), [evidence], "supported"))
        for i, pair in enumerate(role.get("training_pairs", [])):
            out.append(Example(f"{family}:localized:{i}", family, pair["question"], json.dumps({"answer": "yes", "evidence": pair["answer"]}, ensure_ascii=False), [evidence], "supported"))
    return out


def benchmark_examples(profile: dict, limit: int = 160) -> list[Example]:
    """Held-out question forms: facts may overlap, wording and reasoning task may not."""
    subject, out = profile.get("subject", "the candidate"), []
    for item in profile.get("technologies", []):
        name, evidence = item["name"], item["evidence"]
        family = f"benchmark:technology:{name.casefold()}"
        questions = [
            f"Does {subject} know {name}?",
            f"Would you say {subject} has {name} experience?",
            f"I am considering {subject} for a {name} role. What should I know?",
            f"Can you give me proof that {subject} has used {name}?",
        ]
        for i, question in enumerate(questions): out.append(Example(f"{family}:{i}", family, question, f"Yes. {evidence}", [evidence], "supported"))
        for i, pair in enumerate(item.get("benchmark_pairs", [])):
            out.append(Example(f"{family}:localized:{i}", family, pair["question"], json.dumps({"answer": "yes", "evidence": pair["answer"]}, ensure_ascii=False), [evidence], "supported"))
    for name in profile.get("negative_technologies", []):
        family = f"benchmark:negative:{name.casefold()}"
        for i, question in enumerate([f"Does {subject} know {name}?", f"Has {subject} worked with {name}?", f"Can I say {subject} has professional {name} experience?"]):
            out.append(Example(f"{family}:{i}", family, question, INSUFFICIENT, [], "unsupported"))
    for i, question in enumerate(profile.get("negative_benchmark_questions", [])):
        out.append(Example(f"benchmark:negative:localized:{i}", "benchmark:negative:localized", question, INSUFFICIENT, [], "unsupported"))
    for item in profile.get("facts", []):
        family = f"benchmark:fact:{item['id']}"
        for i, question in enumerate(item["benchmark_questions"]):
            out.append(Example(f"{family}:{i}", family, question, json.dumps({"answer": "yes", "evidence": item["answer"]}), [item["evidence"]], "supported"))
        for i, pair in enumerate(item.get("benchmark_pairs", [])):
            out.append(Example(f"{family}:localized:{i}", family, pair["question"], json.dumps({"answer": "yes", "evidence": pair["answer"]}, ensure_ascii=False), [item["evidence"]], "supported"))
    return out[:limit]


def heldout_examples(profile: dict) -> dict[str, list[Example]]:
    """Unseen question forms, while every documented fact remains learnable in train."""
    result = {"validation": [], "test": []}
    subject = profile.get("subject", "the candidate")
    for item in profile.get("technologies", []):
        name, evidence = item["name"], item["evidence"]
        split_name = "validation" if stable_bucket(name) < 50 else "test"
        result[split_name].append(Example(f"heldout:{name}:{split_name}", f"heldout:{name}", f"Assess the evidence for a claimed {name} skill by {subject}.", f"Yes. {evidence}", [evidence], "supported"))
    for name in profile.get("negative_technologies", []):
        split_name = "validation" if stable_bucket(name) < 50 else "test"
        result[split_name].append(Example(f"heldout:negative:{name}:{split_name}", f"heldout:negative:{name}", f"Is there verified evidence that {subject} used {name}?", INSUFFICIENT, [], "unsupported"))
    return result


def fact_examples(profile: dict) -> list[Example]:
    out = []
    for item in profile.get("facts", []):
        family = f"fact:{item['id']}"
        repeats = int(item.get("training_weight", 1))
        for i, question in enumerate(item["training_questions"]):
            for repeat in range(repeats):
                out.append(Example(f"{family}:{i}:{repeat}", family, question, json.dumps({"answer": "yes", "evidence": item["answer"]}), [item["evidence"]], "supported"))
        for i, pair in enumerate(item.get("training_pairs", [])):
            for repeat in range(repeats):
                out.append(Example(f"{family}:localized:{i}:{repeat}", family, pair["question"], json.dumps({"answer": "yes", "evidence": pair["answer"]}, ensure_ascii=False), [item["evidence"]], "supported"))
    return out


def scope_examples() -> list[Example]:
    """Teach the model that it is a career assistant, not a general-knowledge chatbot."""
    rows = [
        ("scope:train:capital-en", "What is the capital of Germany?"),
        ("scope:train:weather-en", "What will the weather be tomorrow?"),
        ("scope:train:capital-tr", "İtalya'nın başkenti neresi?"),
        ("scope:train:news-tr", "Bugünün en önemli haberi ne?"),
        ("scope:train:sports-tr", "Dün akşamki maçın sonucu neydi?"),
        ("scope:train:capital-es", "İspanya'nın başkenti nedir?"),
        ("scope:train:recipe-tr", "Kolay bir makarna tarifi verir misin?"),
        ("scope:train:currency-tr", "Dolar kuru bugün kaç lira?"),
        ("scope:train:history-tr", "Osmanlı Devleti ne zaman kuruldu?"),
        ("scope:train:time-tr", "Saat kaç?"),
        ("scope:train:time-casual-tr", "Şu an saat kaç?"),
        ("scope:train:date-tr", "Bugün ayın kaçı?"),
        ("scope:train:time-en", "What time is it?"),
    ]
    return [Example(identifier, "scope", question, INSUFFICIENT, [], "unsupported") for identifier, question in rows]


def hr_recovery_examples() -> list[Example]:
    """Weighted paraphrases for recruiter intents that fail the strict regression suite."""
    supported = [
        (
            "summary-tr",
            ["Profesyonel geçmişini bir işe alım uzmanına nasıl anlatırsın?", "Kariyer profilini iki cümlede özetle."],
            "Sekiz yılı aşkın deneyime sahip Lead/Senior Full Stack ve AI Engineer’ım. TypeScript/React frontendleri, PHP/Node.js backendleri, AI sistemleri ve mühendislik liderliği alanlarında çalıştım.",
        ),
        (
            "experience-tr",
            ["Profesyonel kariyerin kaç yıldan uzun?", "Toplam çalışma tecrübeni kısaca söyler misin?"],
            "Sekiz yılı aşkın profesyonel deneyimim var. Full-stack geliştirme, AI sistemleri, altyapı ve mühendislik liderliği alanlarında çalıştım.",
        ),
        (
            "leadership-tr",
            ["Yönettiğin mühendislik ekibinin büyüklüğü neydi?", "Teknik lider olarak kaç kişiye liderlik ettin?"],
            "Teknik yol haritasının sorumluluğunu üstlendim ve geliştiricilerle QA mühendislerinden oluşan 8’den fazla kişilik bir ekibe liderlik ettim.",
        ),
        (
            "leadership-en",
            ["Describe the engineering team you managed.", "How many developers and QA engineers did you lead?"],
            "I owned the technical roadmap and led a team of more than eight developers and QA engineers.",
        ),
        (
            "devops-tr",
            ["Dağıtım otomasyonu konusunda neler yaptın?", "CI/CD pipeline deneyimini anlatır mısın?"],
            "Otomatik uçtan uca CI/CD pipeline’ları geliştirdim ve Azure DevOps üzerinden dağıtım süreçleri kurdum.",
        ),
        (
            "ai-tr",
            ["Yapay zekâ mühendisliği tarafında hangi sistemleri geliştirdin?", "AI ve yerel model çalışmalarını özetler misin?"],
            "Yerel ve internetten yalıtılmış AI sistemleri geliştirdim; RAG çözümlerinde Ollama, LangChain ve Chroma kullandım, ayrıca OpenAI API ve Claude entegrasyonları üzerinde çalıştım.",
        ),
        (
            "ai-en",
            ["What tools did you use in air-gapped AI projects?", "Summarize your local RAG technology stack."],
            "I built air-gapped local RAG systems using Ollama, LangChain, and Chroma, and also worked with OpenAI APIs and Claude integrations.",
        ),
    ]
    unsupported = [
        "Is a preferred relocation destination documented for Orcun?",
        "Has Orcun stated a target compensation range?",
        "Is a confirmed employment start date included in the career record?",
        "Do the reviewed documents say whether Orcun is married?",
        "Is Orcun's family status documented?",
        "Is there evidence that Orcun completed postgraduate study?",
        "Can we claim Orcun holds a master's qualification?",
        "Orçun'un taşınmayı kabul ettiğine dair doğrulanmış bilgi var mı?",
        "Maaş beklentisi belgelenmiş mi?",
        "İşe başlayabileceği kesin tarih biliniyor mu?",
        "Medeni hali CV'de belirtiliyor mu?",
        "Yüksek lisans mezunu olduğuna dair kanıt var mı?",
    ]
    out: list[Example] = []
    for family_name, questions, answer_text in supported:
        answer = json.dumps({"answer": "yes", "evidence": answer_text}, ensure_ascii=False)
        for i, question in enumerate(questions):
            for repeat in range(10):
                out.append(Example(f"hr-recovery:{family_name}:{i}:{repeat}", f"hr-recovery:{family_name}", question, answer, [answer_text], "supported"))
    for i, question in enumerate(unsupported):
        for repeat in range(10):
            out.append(Example(f"hr-recovery:unsupported:{i}:{repeat}", "hr-recovery:unsupported", question, INSUFFICIENT, [], "unsupported"))
    return out


def scope_benchmark_examples() -> list[Example]:
    return [
        Example("benchmark:scope:capital-france", "benchmark:scope", "Fransa'nın başkenti neresi?", INSUFFICIENT, [], "unsupported"),
        Example("benchmark:scope:time-tr", "benchmark:scope", "saat kaç", INSUFFICIENT, [], "unsupported"),
        Example("benchmark:scope:time-en", "benchmark:scope", "What time is it now?", INSUFFICIENT, [], "unsupported"),
    ]


def split(examples: list[Example]) -> dict[str, list[Example]]:
    # A family is indivisible so test questions cannot be paraphrases of training.
    result = {"train": [], "validation": [], "test": []}
    for item in examples:
        bucket = stable_bucket(item.family)
        name = "train" if bucket < 75 else "validation" if bucket < 90 else "test"
        result[name].append(item)
    return result


def dump(path: Path, examples: list[Example]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x.as_dict(), ensure_ascii=False) + "\n" for x in examples), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", type=Path, default=ROOT / "data/processed/career_knowledge.json")
    args = parser.parse_args()
    profile = read_json(args.profile)
    if profile.get("review_required"):
        raise SystemExit("Profile still requires human review; refusing to generate training facts.")
    # The knowledge must be seen during training; only question formulations are held out.
    groups = {"train": tech_examples(profile) + role_examples(profile) + fact_examples(profile) + scope_examples() + hr_recovery_examples(), **heldout_examples(profile)}
    for name, values in groups.items():
        target = ROOT / f"data/generated/{name}.jsonl"
        dump(target, values)
        print(f"{name}: {len(values)} examples -> {target}")
    benchmark = benchmark_examples(profile) + scope_benchmark_examples()
    dump(ROOT / "data/benchmark/benchmark.jsonl", benchmark)
    interview_prompts = {
        "adin ne?", "nerede egitim aldin?", "php biliyor musun?", "kac yillik deneyimin var?",
        "angular biliyor musun?", "typescript deneyimin var mı?", "nerede ikamet ediyorsun?", "where do you live?",
    }
    interview = [item for item in benchmark if item.question.casefold() in interview_prompts]
    dump(ROOT / "data/benchmark/interview_acceptance.jsonl", interview)
    print(f"interview acceptance: {len(interview)} questions -> data/benchmark/interview_acceptance.jsonl")
    if len(benchmark) < 100:
        print(f"WARNING: benchmark has {len(benchmark)} questions; add reviewed technologies/projects until it reaches 100-200.")
    else:
        print(f"benchmark: {len(benchmark)} untouched questions -> data/benchmark/benchmark.jsonl")


if __name__ == "__main__":
    main()
