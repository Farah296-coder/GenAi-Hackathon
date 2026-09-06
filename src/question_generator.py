"""
question_generator.py
----------------------
This is Hend's part of the pipeline (Prompt + Question Generation).

Flow:
    relevant_context (from Habiba's rag_search.py)
        -> build_prompt()          # build a clear instruction prompt + inject context
        -> call_llm()              # send prompt to the LLM
        -> parse_llm_response()    # turn raw LLM text into a fixed JSON structure
        -> generate_question()     # ties it all together, returns final dict for Nima

Run this file directly to see a demo using a fake/mock context (no API key needed),
or plug in rag_search.search_question() to use real retrieved context.
"""

import json
import os

# -----------------------------
# 1. Config: allowed question types
# -----------------------------

QUESTION_TYPES = ["MCQ", "True-False", "Short Answer"]


# -----------------------------
# 2. Prompt Template Builder
# -----------------------------

def build_prompt(context: str, question_type: str, topic_hint: str = "") -> str:
    """
    Builds a clear instruction prompt for the LLM, with the retrieved
    context embedded inside it, and strict output-format instructions
    so the LLM's answer is easy to parse later.
    """
    if question_type not in QUESTION_TYPES:
        raise ValueError(f"question_type must be one of {QUESTION_TYPES}")

    base_instructions = f"""You are an assistant that writes exam questions for a teacher.
Use ONLY the information in the CONTEXT below. Do not invent facts that are not in it.
Question type required: {question_type}
{"Focus specifically on: " + topic_hint if topic_hint else ""}

Respond with STRICT JSON ONLY (no extra text, no markdown fences), matching exactly
this shape for the requested question type:
"""

    if question_type == "MCQ":
        shape = """{
  "question": "string",
  "choices": ["string", "string", "string", "string"],
  "correct_answer": "string (must exactly match one of the choices)"
}"""
    elif question_type == "True-False":
        shape = """{
  "question": "string",
  "choices": ["True", "False"],
  "correct_answer": "True or False"
}"""
    else:  # Short Answer
        shape = """{
  "question": "string",
  "correct_answer": "string (a short, direct answer)"
}"""

    prompt = f"""{base_instructions}
{shape}

CONTEXT:
\"\"\"
{context}
\"\"\"
"""
    return prompt


# -----------------------------
# 3. LLM Call
# -----------------------------

from openai import OpenAI

# Groq's API is free and OpenAI-compatible, so we just point the OpenAI
# client at Groq's base_url and use a Groq API key instead.
_client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)

# Free-tier-accessible model on Groq (Llama models there are now
# Enterprise-only). GPT-OSS 20B is fast and works well for structured JSON.
LLM_MODEL = "openai/gpt-oss-20b"


def call_llm(prompt: str) -> str:
    """
    Sends the prompt to Groq's chat completions endpoint and returns
    the raw text response (expected to be a JSON string, per the
    instructions baked into build_prompt()).
    """
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You always respond with strict, valid JSON only. No markdown fences, no extra commentary.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )
    return response.choices[0].message.content


# -----------------------------
# 4. Parse + Format Output
# -----------------------------

def parse_llm_response(raw_output: str, question_type: str, source_page=None) -> dict:
    """
    Parses the LLM's raw text into the fixed structure that Nima expects.
    Raises a clear error if the LLM didn't return valid JSON, so bad
    output never silently gets passed downstream.
    """
    try:
        data = json.loads(raw_output.strip())
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON: {e}\nRaw output:\n{raw_output}")

    structured = {
        "question_type": question_type,
        "question": data.get("question"),
        "correct_answer": data.get("correct_answer"),
    }

    if question_type in ("MCQ", "True-False"):
        structured["choices"] = data.get("choices")

    if source_page is not None:
        structured["source_page"] = source_page

    return structured


# -----------------------------
# 5. Full pipeline for one chunk of context
# -----------------------------

def generate_question(context: str, question_type: str, source_page=None, topic_hint: str = "") -> dict:
    prompt = build_prompt(context, question_type, topic_hint)
    raw_output = call_llm(prompt)
    return parse_llm_response(raw_output, question_type, source_page)


# -----------------------------
# 6. Demo / manual test
# -----------------------------

def generate_from_topic(topic: str, question_type: str, top_k: int = 1):
    """
    Full end-to-end helper: pulls context from Habiba's rag_search.py for
    a given topic, then generates a question from the top matching chunk.
    Requires vector_db/ to already exist (i.e. rag_indexer.py has been run).
    """
    from rag_search import search_question  # Habiba's module, must be on the path

    results = search_question(topic, top_k=top_k)
    context = results["documents"][0][0]
    page = results["metadatas"][0][0]["page"]

    return generate_question(context, question_type, source_page=page, topic_hint=topic)


# -----------------------------
# 7. Generate a whole quiz at once
# -----------------------------

def generate_quiz(requests: list) -> list:
    """
    Generates several questions in one call, ready to hand off to Nima
    as a single list.

    `requests` is a list of dicts, each describing one question to generate:
        [
            {"topic": "generic classes", "question_type": "MCQ"},
            {"topic": "generic classes", "question_type": "True-False"},
            {"topic": "method overriding", "question_type": "Short Answer"},
        ]

    Returns a list of structured question dicts (same shape as
    generate_question()'s output). If one question fails to generate
    (bad JSON from the LLM, topic not found, etc.), it's skipped and
    logged instead of crashing the whole batch.
    """
    quiz = []

    for i, req in enumerate(requests, start=1):
        topic = req["topic"]
        question_type = req["question_type"]

        print(f"[{i}/{len(requests)}] Generating {question_type} question on '{topic}'...")

        try:
            question = generate_from_topic(topic, question_type)
            quiz.append(question)
        except Exception as e:
            print(f"  Skipped — failed to generate this one: {e}")

    return quiz


def save_quiz(quiz: list, path: str = "output/generated_quiz.json"):
    """
    Saves the generated quiz to a JSON file, e.g. for Nima to pick up.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(quiz, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(quiz)} questions to {path}")


if __name__ == "__main__":
    # Quick manual test using a mock context — no vector_db or API key needed
    # just to check the prompt looks right:
    mock_context = (
        "A generic class is a class that is parameterized over types. "
        "Generics allow classes and methods to operate on objects of "
        "various types while providing compile-time type safety."
    )
    prompt = build_prompt(mock_context, "MCQ", topic_hint="generic classes")
    print("----- GENERATED PROMPT -----")
    print(prompt)

    # Full real run (needs OPENAI_API_KEY set + mock_context replaced by
    # real retrieved context, or use generate_from_topic() below instead):
    #
    # question = generate_question(mock_context, "MCQ", source_page=1)
    # print(json.dumps(question, indent=2, ensure_ascii=False))
    #
    # Or, wired directly into Habiba's retrieval, one question at a time:
    #
    question = generate_from_topic("generic classes and methods", "MCQ")
    print(json.dumps(question, indent=2, ensure_ascii=False))

    # Or generate a whole quiz at once and save it for Nima:
    #
    # quiz_requests = [
    #     {"topic": "generic classes", "question_type": "MCQ"},
    #     {"topic": "generic classes", "question_type": "True-False"},
    #     {"topic": "generic methods", "question_type": "Short Answer"},
    # ]
    # quiz = generate_quiz(quiz_requests)
    # save_quiz(quiz)
