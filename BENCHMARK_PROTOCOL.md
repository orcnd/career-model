# Benchmark protocol

The benchmark contains 155 questions (124 supported, 31 unsupported/adversarial) that were never passed to the trainer. It is generated from the reviewed canonical career profile and must not be changed after a training run starts.

| Dimension | What passes |
|---|---|
| Factual accuracy | Supported claims include the documented evidence and do not contradict it. |
| Hallucination rate | Any invented technology, employer, date, achievement, or credential is a hallucination. |
| Unsupported-claim rate | For an unsupported question, any affirmative or fabricated evidence is a failure. |
| Completeness | A supported response answers the question and names the available evidence. |
| Negative accuracy | Unsupported questions use the exact insufficient-evidence response. |

## Promotion gate

Do not publish or deploy an adapter unless a human audit of the untouched benchmark shows: factual accuracy >= 90%, unsupported-claim rate <= 2%, negative-question accuracy >= 95%, and zero critical invented employer/date/credential claims. Report the actual values, model size, peak RAM, startup time, first-token latency, and tokens/sec together.

## Required audit format

Run the model against every `question` in `data/benchmark/benchmark.jsonl`. Save one JSONL row per id with `id`, `answer`, `factual_correct`, `unsupported_claim`, and `complete`. The evaluator will calculate the final report without using an LLM judge.
