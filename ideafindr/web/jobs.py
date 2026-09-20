"""Background research jobs for the web UI.

The pipeline is synchronous and owns its own event loop (`asyncio.run` inside
`collect`), so a job runs in a worker thread and the HTTP layer only ever reads
the shared status dict. That keeps a multi-minute research run off the request
path and out of the server's event loop.

One process, in-memory state: this is a single-user tool on one machine, so a
job registry keyed by id is enough. A restart forgets running jobs; anything
already written to SQLite is still listed under Runs.
"""

from __future__ import annotations

import logging
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

log = logging.getLogger(__name__)

# The stages the full research pipeline runs, in order. Actions (re-analyse,
# harvest demand) declare their own shorter stage lists.
RESEARCH_STAGES = [
    ("plan", "Plan the sweep"),
    ("collect", "Collect the corpus"),
    ("analyze", "Cluster into themes"),
    ("report", "Write the report"),
]


@dataclass
class Job:
    id: str
    topic: str
    days: int
    sources: list[str]
    kind: str = "research"  # research | analyze | demand | report
    title: str = ""
    stages: list[tuple[str, str]] = field(default_factory=lambda: list(RESEARCH_STAGES))
    status: str = "running"  # running | done | failed
    stage: str = ""
    stage_states: dict[str, str] = field(default_factory=dict)
    notes: dict[str, str] = field(default_factory=dict)
    logs: list[tuple[str, str]] = field(default_factory=list)
    run_id: str | None = None
    error: str = ""
    finished: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        if not self.stage_states:
            self.stage_states = {k: "todo" for k, _ in self.stages}
        if not self.stage:
            self.stage = self.stages[0][0] if self.stages else ""

    def log_line(self, message: str, level: str = "info") -> None:
        with self.lock:
            stamp = datetime.now().strftime("%H:%M:%S")
            self.logs.append((stamp, message))
            if len(self.logs) > 400:
                del self.logs[:200]

    def set_stage(self, key: str, state: str, note: str = "") -> None:
        with self.lock:
            self.stage = key
            self.stage_states[key] = state
            if note:
                self.notes[key] = note

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "id": self.id,
                "topic": self.topic,
                "kind": self.kind,
                "title": self.title or self.topic,
                "stages": list(self.stages),
                "status": self.status,
                "stage": self.stage,
                "stage_states": dict(self.stage_states),
                "notes": dict(self.notes),
                "logs": list(self.logs),
                "run_id": self.run_id,
                "error": self.error,
                "finished": self.finished,
            }


class JobRegistry:
    """Thread-safe map of id -> Job, with a bounded history."""

    def __init__(self, keep: int = 50) -> None:
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._keep = keep
        self._lock = threading.Lock()

    def create(self, topic: str, days: int, sources: list[str], **kwargs) -> Job:
        job = Job(id=uuid.uuid4().hex[:12], topic=topic, days=days, sources=sources, **kwargs)
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            while len(self._order) > self._keep:
                self._jobs.pop(self._order.pop(0), None)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def latest(self) -> Job | None:
        with self._lock:
            return self._jobs[self._order[-1]] if self._order else None

    def active(self) -> list[Job]:
        with self._lock:
            return [self._jobs[i] for i in self._order if self._jobs[i].status == "running"]


registry = JobRegistry()


def _start(job: Job, target: Callable[[Job], None]) -> Job:
    t = threading.Thread(target=_run, args=(job, target), daemon=True,
                         name=f"{job.kind}-{job.id}")
    t.start()
    return job


def start_research(topic: str, days: int, sources: list[str]) -> Job:
    """Kick off a full research run in a background thread and return the job."""
    job = registry.create(topic, days, sources, kind="research", title=topic)
    return _start(job, _run_research)


def start_action(run_id: str, topic: str, kind: str) -> Job:
    """Re-run one stage over an already-collected corpus (analyse or demand)."""
    stages = {
        "analyze": [("analyze", "Cluster and label themes")],
        "demand": [("demand", "Harvest search demand")],
    }[kind]
    job = registry.create(topic, 0, [], kind=kind, title=f"{kind}: {topic}",
                          stages=stages, run_id=run_id)
    target = _run_analyze if kind == "analyze" else _run_demand
    return _start(job, target)


class _LogBridge(logging.Handler):
    """Mirror pipeline log records into the job's log so the UI shows what the
    CLI would have printed."""

    def __init__(self, job: Job) -> None:
        super().__init__(level=logging.INFO)
        self._job = job

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = "warn" if record.levelno >= logging.WARNING else "info"
            self._job.log_line(record.getMessage(), level)
        except Exception:  # noqa: BLE001
            pass


def _run(job: Job, target: Callable[[Job], None]) -> None:
    handler = _LogBridge(job)
    root = logging.getLogger("ideafindr")
    root.addHandler(handler)
    try:
        target(job)
        job.status = "done"
    except Exception as e:  # noqa: BLE001 - a failed job reports, never crashes the server
        job.status = "failed"
        job.error = str(e)
        job.log_line(f"failed: {e}", "warn")
        log.warning("job %s failed: %s\n%s", job.id, e, traceback.format_exc())
    finally:
        job.finished = True
        root.removeHandler(handler)


def _run_research(job: Job) -> None:
    from ideafindr import pipeline

    job.set_stage("plan", "active")
    plan = pipeline.plan_for(job.topic, job.days)
    job.set_stage("plan", "done",
                  f"{len(plan.subreddits)} communities, {len(plan.keywords)} keywords")

    job.set_stage("collect", "active")
    run_id, n = pipeline.run_collection(job.topic, job.sources, job.days, 600, plan=plan)
    job.run_id = run_id
    if not n:
        job.set_stage("collect", "failed", "nothing collected")
        raise RuntimeError("no documents collected; try a broader topic or more sources")
    job.set_stage("collect", "done", f"{n} documents")

    job.set_stage("analyze", "active")
    themes = pipeline.run_analysis(run_id)
    job.set_stage("analyze", "done", f"{len(themes)} themes")

    job.set_stage("report", "active")
    pipeline.render_report(run_id)
    job.set_stage("report", "done", "report ready")


def _run_analyze(job: Job) -> None:
    from ideafindr import pipeline

    job.set_stage("analyze", "active")
    themes = pipeline.run_analysis(job.run_id)
    job.set_stage("analyze", "done", f"{len(themes)} themes")


def _run_demand(job: Job) -> None:
    from ideafindr import pipeline

    job.set_stage("demand", "active")
    clusters = pipeline.run_demand(job.run_id, label=True)
    job.set_stage("demand", "done", f"{len(clusters)} intents")

