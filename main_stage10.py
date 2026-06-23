"""
Stage 10 — Security + Guardrails (minimal rebuild)
One file. No global mutable state shared across runs.
Every orchestration run gets its own run_id, and all approval
state lives inside a dict keyed by that run_id — so there is
no possible way for one run's state to leak into another.
"""

import re
import json
import uuid
import asyncio
from dataclasses import dataclass, field

import ollama
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# ----------------------------------------------------------------------
# GUARDRAILS
# ----------------------------------------------------------------------

INJECTION_PATTERNS = [
    r"ignore (all )?(the |your )?(previous|prior|above)? ?instructions?",
    r"disregard (all )?(the |your )?(previous|prior|above)? ?instructions?",
    r"forget (everything|all|what)( you| that)?",
    r"you are now",
    r"system prompt",
    r"reveal (your|the) (instructions|prompt|system prompt|secrets?)",
    r"give me (the|your) (secrets?|password|api key|credentials)",
    r"act as if",
    r"new instructions:",
    r"pretend (you|that)",
    r"no longer (bound|restricted|limited) by",
    r"bypass (your|the) (rules|restrictions|guardrails|safety)",
]

PII_PATTERNS = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"\b(\+?\d{1,2}[\s-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
}


def check_input(text: str) -> tuple[bool, str]:
    """Returns (safe, reason)."""
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return False, f"Possible prompt injection (matched: '{pattern}')"
    return True, ""


def check_output(text: str) -> tuple[str, list[str]]:
    """Returns (cleaned_text, pii_labels_found). Does not block, only redacts."""
    found = []
    cleaned = text
    for label, pattern in PII_PATTERNS.items():
        if re.search(pattern, cleaned):
            found.append(label)
            cleaned = re.sub(pattern, f"[REDACTED_{label.upper()}]", cleaned)
    return cleaned, found


# ----------------------------------------------------------------------
# APPROVAL GATE — run-scoped, not global
# ----------------------------------------------------------------------

@dataclass
class RunState:
    events: dict[str, asyncio.Event] = field(default_factory=dict)
    decisions: dict[str, bool] = field(default_factory=dict)


class ApprovalGate:
    def __init__(self):
        self._runs: dict[str, RunState] = {}

    def start_run(self, run_id: str):
        self._runs[run_id] = RunState()

    def register(self, run_id: str, agent_name: str):
        run = self._runs[run_id]
        run.events[agent_name] = asyncio.Event()
        run.decisions[agent_name] = False

    def decide(self, run_id: str, agent_name: str, approved: bool):
        run = self._runs.get(run_id)
        if not run or agent_name not in run.events:
            return False
        run.decisions[agent_name] = approved
        run.events[agent_name].set()
        return True

    async def wait_for_approval(self, run_id: str, agent_name: str, timeout: float = 300.0) -> bool:
        run = self._runs[run_id]
        event = run.events[agent_name]
        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
            return run.decisions.get(agent_name, False)
        except asyncio.TimeoutError:
            return False

    def end_run(self, run_id: str):
        self._runs.pop(run_id, None)


gate = ApprovalGate()

# ----------------------------------------------------------------------
# "AGENTS" — just direct Ollama calls, kept dumb and simple on purpose
# ----------------------------------------------------------------------

def call_llm(system: str, prompt: str) -> str:
    response = ollama.chat(
        model="llama3.1",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
    )
    return response["message"]["content"]


async def run_research(query: str) -> str:
    return await asyncio.to_thread(
        call_llm,
        "You are a research assistant. Give 3-4 concise factual bullet points.",
        query,
    )


async def run_writer(query: str, research_notes: str) -> str:
    return await asyncio.to_thread(
        call_llm,
        "You are a writer. Turn research notes into a short, clear paragraph for the user.",
        f"Query: {query}\n\nResearch notes:\n{research_notes}",
    )


# ----------------------------------------------------------------------
# FASTAPI APP
# ----------------------------------------------------------------------

app = FastAPI(title="Stage 10 — Minimal Rebuild")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def sse(data: dict) -> str:
    return f"data: {json.dumps(data)}\n\n"


@app.get("/orchestrate/stream")
async def orchestrate_stream(query: str):
    run_id = str(uuid.uuid4())

    async def stream():
        # 🛡️ input guardrail
        safe, reason = check_input(query)
        if not safe:
            yield sse({"run_id": run_id, "agent": "guardrail", "status": "blocked", "msg": reason})
            return

        gate.start_run(run_id)
        yield sse({"run_id": run_id, "agent": "supervisor", "status": "planning", "msg": f"Planning for: {query}"})

        pipeline = ["research", "writer"]
        research_notes = ""

        for agent_name in pipeline:
            gate.register(run_id, agent_name)
            yield sse({"run_id": run_id, "agent": agent_name, "status": "awaiting_approval", "msg": f"Run {agent_name}?"})

            approved = await gate.wait_for_approval(run_id, agent_name)
            if not approved:
                yield sse({"run_id": run_id, "agent": agent_name, "status": "skipped", "msg": "Rejected by user"})
                continue

            yield sse({"run_id": run_id, "agent": agent_name, "status": "started", "msg": "Working..."})

            if agent_name == "research":
                research_notes = await run_research(query)
                yield sse({"run_id": run_id, "agent": agent_name, "status": "done", "msg": research_notes[:200]})
            elif agent_name == "writer":
                final_text = await run_writer(query, research_notes)
                cleaned, pii_found = check_output(final_text)
                yield sse({
                    "run_id": run_id,
                    "agent": "writer",
                    "status": "done",
                    "msg": cleaned[:200],
                    "pii_found": pii_found,
                })

        gate.end_run(run_id)
        yield sse({"run_id": run_id, "agent": "supervisor", "status": "final", "msg": "Run complete."})

    return StreamingResponse(stream(), media_type="text/event-stream")


@app.post("/approve/{run_id}/{agent_name}")
async def approve(run_id: str, agent_name: str):
    ok = gate.decide(run_id, agent_name, approved=True)
    return {"ok": ok}


@app.post("/reject/{run_id}/{agent_name}")
async def reject(run_id: str, agent_name: str):
    ok = gate.decide(run_id, agent_name, approved=False)
    return {"ok": ok}