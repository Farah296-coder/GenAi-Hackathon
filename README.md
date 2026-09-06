# Prompt + Question Generation

This module covers the **Prompt + Question Generation** part of the pipeline: turning retrieved context into a structured, ready-to-use exam question.

## Where it fits in the pipeline

## File

- `src/question_generator.py`

## Setup

Install dependencies:

```bash
pip install openai chromadb sentence-transformers pymupdf
```

This module uses **Groq** (free, OpenAI-compatible API) as the LLM provider. You'll need a free API key from [console.groq.com/keys](https://console.groq.com/keys).

Set it as an environment variable:

```bash
# macOS / Linux
export GROQ_API_KEY="gsk_..."

# Windows (PowerShell / cmd)
setx GROQ_API_KEY "gsk_..."
```

> Model in use: `openai/gpt-oss-20b`.

## How it works

| Function | What it does |
|---|---|
| `build_prompt(context, question_type, topic_hint)` | Builds the instruction prompt sent to the LLM, embedding the retrieved context and specifying the exact JSON shape expected back. |
| `call_llm(prompt)` | Sends the prompt to Groq and returns the raw text response. |
| `parse_llm_response(raw_output, question_type, source_page)` | Parses the LLM's response into a fixed dict structure. Raises a clear error if the LLM didn't return valid JSON. |
| `generate_question(context, question_type, source_page, topic_hint)` | Runs the full pipeline for one piece of context: prompt → LLM → parsed output. |
| `generate_from_topic(topic, question_type, top_k)` | Same as above, but pulls the context automatically from Habiba's `rag_search.search_question()` given just a topic string. |
| `generate_quiz(requests)` | Generates multiple questions at once from a list of `{topic, question_type}` requests. Skips and logs any individual question that fails instead of crashing the whole batch. |
| `save_quiz(quiz, path)` | Saves a generated quiz to a JSON file for handoff. |

Supported question types: `MCQ`, `True-False`, `Short Answer`.

## Example usage

Generate one question:

```python
from question_generator import generate_from_topic

question = generate_from_topic("generic classes and methods", "MCQ")
print(question)
```

Generate a full quiz and save it:

```python
from question_generator import generate_quiz, save_quiz

quiz_requests = [
    {"topic": "generic classes", "question_type": "MCQ"},
    {"topic": "generic classes", "question_type": "True-False"},
    {"topic": "generic methods", "question_type": "Short Answer"},
]

quiz = generate_quiz(quiz_requests)
save_quiz(quiz)  # -> output/generated_quiz.json
```

## Output format

Every generated question follows this structure:

**MCQ**
```json
{
  "question_type": "MCQ",
  "question": "What is a generic class?",
  "choices": ["...", "...", "...", "..."],
  "correct_answer": "...",
  "source_page": 2
}
```

**True-False**
```json
{
  "question_type": "True-False",
  "question": "Generics provide compile-time type safety.",
  "choices": ["True", "False"],
  "correct_answer": "True",
  "source_page": 2
}
```

**Short Answer**
```json
{
  "question_type": "Short Answer",
  "question": "What do generics allow a class to operate on?",
  "correct_answer": "Objects of various types",
  "source_page": 2
}
```

## Notes

- `correct_answer` for MCQ/True-False always matches one of the entries in `choices` exactly, so it can be used directly for automated grading.
- Groq's free tier has rate limits — if you hit a `rate_limit_exceeded` error while batch-generating a quiz, wait a bit and retry.
