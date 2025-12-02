import os
import datetime
from dataclasses import dataclass
from dotenv import load_dotenv
from openai import OpenAI

# -----------------------------
# Data Structure
# -----------------------------
@dataclass
class AssignmentPlan:
    summary: str
    task_plan: str
    schedule: str


# -----------------------------
# Init OpenRouter Client
# -----------------------------
def init_openrouter():
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-70b-instruct")

    if not api_key:
        raise ValueError("OPENROUTER_API_KEY missing in .env")

    client = OpenAI(
        api_key=api_key,
        base_url="https://openrouter.ai/api/v1"
    )

    return client, model


# -----------------------------
# Chat helper
# -----------------------------
def ask_model(client, model, system_prompt, user_prompt):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )

    # OpenRouter uses .message.content, not ["content"]
    return response.choices[0].message.content


# -----------------------------
# Functions for agent
# -----------------------------
def summarize_assignment(client, model, full_text):
    system = "You summarize assignments clearly and concisely."

    user = f"Summarize the following assignment:\n\n{full_text}"
    return ask_model(client, model, system, user)


def plan_tasks(client, model, summary):
    system = "You break academic assignments into clear tasks with time estimates."

    user = f"Using this summary, create a detailed task breakdown:\n\n{summary}"
    return ask_model(client, model, system, user)


def build_schedule(client, model, summary, tasks, due_date):
    system = "You build realistic schedules for students based on tasks and deadlines."

    today = datetime.date.today().isoformat()
    due = due_date.isoformat()

    user = f"""
Today's date: {today}
Due date: {due}

Assignment Summary:
{summary}

Task List:
{tasks}

Create a detailed daily schedule.
"""
    return ask_model(client, model, system, user)


# -----------------------------
# Pipeline
# -----------------------------
def run_agent(full_assignment_text, due_date):
    client, model = init_openrouter()

    print("🔍 Summarizing assignment...\n")
    summary = summarize_assignment(client, model, full_assignment_text)

    print("📝 Creating task plan...\n")
    task_plan = plan_tasks(client, model, summary)

    print("📅 Generating schedule...\n")
    schedule = build_schedule(client, model, summary, task_plan, due_date)

    return AssignmentPlan(summary, task_plan, schedule)


# -----------------------------
# Example Assignment
# -----------------------------
EXAMPLE_ASSIGNMENT_TEXT = """
CapStone Final: Build and Submit an AI Agent for the AI Agents Hackathon 2025.
Course: ITAI 2277.

Your tasks: design, build, and submit an AI Agent using Semantic Kernel.
Submit a GitHub repo, demo video, and reflection report.
Due May 3, 2025.
"""

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    due = datetime.date(2025, 5, 3)
    result = run_agent(EXAMPLE_ASSIGNMENT_TEXT, due)

    print("\n===== SUMMARY =====\n")
    print(result.summary)

    print("\n===== TASK PLAN =====\n")
    print(result.task_plan)

    print("\n===== SCHEDULE =====\n")
    print(result.schedule)