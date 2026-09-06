import json
import os
from openai import OpenAI

PASS_THRESHOLD = 0.7
WEIGHTS = {"correctness": 0.4, "groundedness": 0.3, "clarity": 0.2, "distractor_quality": 0.1}

_client = OpenAI(
    api_key=os.environ["GROQ_API_KEY"],
    base_url="https://api.groq.com/openai/v1",
)
LLM_MODEL = "openai/gpt-oss-20b"


def load_questions(path="output/generated_quiz.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_source_pages(path="output/extracted_text.json"):
    with open(path, "r", encoding="utf-8") as f:
        pages = json.load(f)
    return {p["page"]: p["text"] for p in pages}


def get_source_text(source_page, pages_by_number):
    return pages_by_number.get(source_page, "")


def build_eval_prompt(q, source_text):
    choices_block = f"\nChoices: {q.get('choices')}" if q.get("choices") else ""
    return f"""You are a strict exam-quality reviewer. Judge the QUESTION using ONLY the SOURCE TEXT.
Score each from 0.0 to 1.0:
1. correctness: is the correct_answer actually correct based on the source text?
2. groundedness: is the question's content actually present in the source text?
3. clarity: is the question clearly worded?

QUESTION TYPE: {q.get('question_type')}
QUESTION: {q.get('question')}{choices_block}
MARKED CORRECT ANSWER: {q.get('correct_answer')}

SOURCE TEXT:
\"\"\"
{source_text}
\"\"\"

Respond with STRICT JSON ONLY:
{{
  "correctness": 0.0,
  "groundedness": 0.0,
  "clarity": 0.0,
  "feedback": "short explanation"
}}
"""


def call_llm(prompt):
    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You always respond with strict, valid JSON only."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )
    return response.choices[0].message.content


def parse_eval_response(raw_output):
    data = json.loads(raw_output.strip())
    return {
        "correctness": float(data.get("correctness", 0.0)),
        "groundedness": float(data.get("groundedness", 0.0)),
        "clarity": float(data.get("clarity", 0.0)),
        "feedback": data.get("feedback", ""),
    }


def check_mcq_rules(q):
    issues = []
    if q.get("question_type") != "MCQ":
        return {"distractor_quality": None, "issues": issues}

    choices = q.get("choices") or []
    correct_answer = q.get("correct_answer")

    if correct_answer not in choices:
        issues.append("correct_answer not in choices")
    if len(choices) != len(set(choices)):
        issues.append("duplicate choices")

    distractor_quality = 1.0
    if choices and correct_answer in choices:
        lengths = [len(c) for c in choices]
        avg_length = sum(lengths) / len(lengths)
        if avg_length > 0 and abs(len(correct_answer) - avg_length) > avg_length * 0.75:
            issues.append("correct answer length stands out")
            distractor_quality = 0.5

    if issues:
        distractor_quality = min(distractor_quality, 0.5)

    return {"distractor_quality": distractor_quality, "issues": issues}


def score_question(judge_scores, mcq_check):
    is_mcq = mcq_check["distractor_quality"] is not None
    if is_mcq:
        overall = (
            judge_scores["correctness"] * WEIGHTS["correctness"]
            + judge_scores["groundedness"] * WEIGHTS["groundedness"]
            + judge_scores["clarity"] * WEIGHTS["clarity"]
            + mcq_check["distractor_quality"] * WEIGHTS["distractor_quality"]
        )
    else:
        remaining = WEIGHTS["correctness"] + WEIGHTS["groundedness"] + WEIGHTS["clarity"]
        overall = (
            judge_scores["correctness"] * (WEIGHTS["correctness"] / remaining)
            + judge_scores["groundedness"] * (WEIGHTS["groundedness"] / remaining)
            + judge_scores["clarity"] * (WEIGHTS["clarity"] / remaining)
        )
    return {"overall_score": round(overall, 3), "passed": overall >= PASS_THRESHOLD}


def evaluate_question(q, pages_by_number):
    source_page = q.get("source_page")
    source_text = get_source_text(source_page, pages_by_number)

    prompt = build_eval_prompt(q, source_text)
    raw_output = call_llm(prompt)
    judge_scores = parse_eval_response(raw_output)

    mcq_check = check_mcq_rules(q)
    final = score_question(judge_scores, mcq_check)

    return {
        "question": q.get("question"),
        "question_type": q.get("question_type"),
        "source_page": source_page,
        "judge_scores": {
            "correctness": judge_scores["correctness"],
            "groundedness": judge_scores["groundedness"],
            "clarity": judge_scores["clarity"],
        },
        "judge_feedback": judge_scores["feedback"],
        "mcq_issues": mcq_check["issues"],
        "overall_score": final["overall_score"],
        "passed": final["passed"],
    }


def evaluate_quiz(questions, pages_by_number):
    results = []
    for q in questions:
        try:
            results.append(evaluate_question(q, pages_by_number))
        except Exception as e:
            print(f"Skipped: {e}")
    return results


def save_results(results, path="output/evaluation_results.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(results)} results to {path}")


def ensure_quiz_exists(path="output/generated_quiz.json"):
    if os.path.exists(path):
        return

    from question_generator import generate_quiz, save_quiz

    quiz_requests = [
        {"topic": "generic classes", "question_type": "MCQ"},
        {"topic": "generic classes", "question_type": "True-False"},
        {"topic": "generic methods", "question_type": "Short Answer"},
    ]
    quiz = generate_quiz(quiz_requests)
    save_quiz(quiz, path)


if __name__ == "__main__":
    ensure_quiz_exists()

    questions = load_questions("output/generated_quiz.json")
    pages_by_number = load_source_pages("output/extracted_text.json")
    results = evaluate_quiz(questions, pages_by_number)
    save_results(results)