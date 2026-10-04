"""SQLite and JSONL storage for traced sessions, turns, and failure annotations."""

from __future__ import annotations

import json
import sqlite3
import uuid
from pathlib import Path
from typing import Any

from traced_agent.models import Annotation, Step, StepKind, TaxonomyCategory, Trace, Turn, Verdict

DEFAULT_TAXONOMY: list[TaxonomyCategory] = [
    TaxonomyCategory(
        key="wrong_tool",
        name="Wrong Tool Selection",
        description="Agent selected an inappropriate tool or bypassed a dedicated tool in favor of generic queries/raw commands.",
        examples=["Used find/grep instead of archive_game_logs", "Used search_foxhole_wiki instead of get_item_stats"],
    ),
    TaxonomyCategory(
        key="hallucinated_ui",
        name="Hallucinated Context / UI",
        description="Agent invented nonexistent UI menus, interactions, buttons, or context actions.",
        examples=["Advised selecting 'Retrieve as Item' in Seaport stockpile"],
    ),
    TaxonomyCategory(
        key="topological_inversion",
        name="Topological / Spatial Inversion",
        description="Agent inverted spatial, geographic, or waterway topology instead of following actual map corridors.",
        examples=["Assumed river entered hex from west rather than curved south entry", "Suggested destination past target as intermediate stop"],
    ),
    TaxonomyCategory(
        key="anchoring_drift",
        name="Multi-Turn Anchoring / Drift",
        description="Agent committed early to an incorrect assumption and failed to revise it across subsequent turns.",
        examples=["Persisted in trying to uncrate at Seaport after player corrected the facility type"],
    ),
    TaxonomyCategory(
        key="missing_constraint",
        name="Missing Constraint",
        description="Agent dropped or ignored explicit constraints specified in the prompt or ground truth context.",
        examples=["Forgot pet-friendly filter", "Ignored vehicle crate packaging requirement"],
    ),
]


class TraceStorage:
    def __init__(self, db_path: Path | str = "traces.db") -> None:
        self.db_path = Path(db_path)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    agent_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    data_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS annotations (
                    annotation_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    turn_index INTEGER NOT NULL,
                    verdict TEXT NOT NULL,
                    first_failure_step_index INTEGER,
                    open_code TEXT,
                    failure_modes_json TEXT NOT NULL,
                    annotator TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(trace_id) REFERENCES traces(trace_id)
                );

                CREATE TABLE IF NOT EXISTS taxonomy (
                    key TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    examples_json TEXT NOT NULL
                );
            """)
            # Populate default taxonomy if empty
            cursor = conn.execute("SELECT COUNT(*) FROM taxonomy")
            if cursor.fetchone()[0] == 0:
                for cat in DEFAULT_TAXONOMY:
                    conn.execute(
                        "INSERT INTO taxonomy (key, name, description, examples_json) VALUES (?, ?, ?, ?)",
                        (cat.key, cat.name, cat.description, json.dumps(cat.examples)),
                    )
            conn.commit()

    def get_taxonomy(self) -> list[TaxonomyCategory]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT key, name, description, examples_json FROM taxonomy").fetchall()
            return [
                TaxonomyCategory(
                    key=row["key"],
                    name=row["name"],
                    description=row["description"],
                    examples=json.loads(row["examples_json"]),
                )
                for row in rows
            ]

    def save_trace(self, trace: Trace) -> None:
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces (trace_id, session_id, agent_name, created_at, data_json) VALUES (?, ?, ?, ?, ?)",
                (
                    trace.trace_id,
                    trace.session_id,
                    trace.agent_name,
                    trace.created_at.isoformat(),
                    trace.model_dump_json(),
                ),
            )
            conn.commit()

    def get_trace(self, trace_id: str) -> Trace | None:
        with self._get_conn() as conn:
            row = conn.execute("SELECT data_json FROM traces WHERE trace_id = ?", (trace_id,)).fetchone()
            if not row:
                return None
            return Trace.model_validate_json(row["data_json"])

    def list_traces(self) -> list[Trace]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT data_json FROM traces ORDER BY created_at DESC").fetchall()
            return [Trace.model_validate_json(r["data_json"]) for r in rows]

    def save_annotation(self, ann: Annotation) -> None:
        with self._get_conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO annotations (
                    annotation_id, trace_id, turn_index, verdict,
                    first_failure_step_index, open_code, failure_modes_json, annotator, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ann.annotation_id,
                    ann.trace_id,
                    ann.turn_index,
                    ann.verdict.value,
                    ann.first_failure_step_index,
                    ann.open_code,
                    json.dumps(ann.failure_modes),
                    ann.annotator,
                    ann.created_at.isoformat(),
                ),
            )
            conn.commit()

    def get_annotations_for_trace(self, trace_id: str) -> list[Annotation]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM annotations WHERE trace_id = ? ORDER BY turn_index ASC", (trace_id,)).fetchall()
            return [
                Annotation(
                    annotation_id=r["annotation_id"],
                    trace_id=r["trace_id"],
                    turn_index=r["turn_index"],
                    verdict=Verdict(r["verdict"]),
                    first_failure_step_index=r["first_failure_step_index"],
                    open_code=r["open_code"],
                    failure_modes=json.loads(r["failure_modes_json"]),
                    annotator=r["annotator"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def list_all_annotations(self) -> list[Annotation]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT * FROM annotations ORDER BY created_at DESC").fetchall()
            return [
                Annotation(
                    annotation_id=r["annotation_id"],
                    trace_id=r["trace_id"],
                    turn_index=r["turn_index"],
                    verdict=Verdict(r["verdict"]),
                    first_failure_step_index=r["first_failure_step_index"],
                    open_code=r["open_code"],
                    failure_modes=json.loads(r["failure_modes_json"]),
                    annotator=r["annotator"],
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    def ingest_sortie_jsonl(self, jsonl_path: Path) -> Trace:
        """Parse a Foxhole sortie jsonl file into a structured multi-turn Trace."""
        turns: list[Turn] = []
        session_id = jsonl_path.stem.replace("sortie_", "")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for turn_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                user_input = data.get("input", "")
                actual_output = data.get("actual_output", "")
                tools = data.get("tools_called", [])

                steps: list[Step] = []
                step_counter = 0

                # 1. User Input Step
                steps.append(Step(step_index=step_counter, kind=StepKind.USER_INPUT, text=user_input))
                step_counter += 1

                # 2. Tool Calls & Outputs
                for t in tools:
                    t_name = t.get("name", "")
                    t_params = t.get("input_parameters", {})
                    t_out = t.get("output", "")
                    steps.append(
                        Step(
                            step_index=step_counter,
                            kind=StepKind.TOOL_CALL,
                            name=t_name,
                            payload=t_params if isinstance(t_params, dict) else {},
                        )
                    )
                    step_counter += 1
                    steps.append(
                        Step(
                            step_index=step_counter,
                            kind=StepKind.TOOL_OUTPUT,
                            name=t_name,
                            text=str(t_out),
                        )
                    )
                    step_counter += 1

                # 3. Final Completion Step
                steps.append(
                    Step(step_index=step_counter, kind=StepKind.COMPLETION, text=actual_output)
                )

                turns.append(
                    Turn(
                        turn_index=turn_idx,
                        input_text=user_input,
                        output_text=actual_output,
                        steps=steps,
                    )
                )

        trace = Trace(
            trace_id=uuid.uuid4().hex[:12],
            session_id=session_id,
            agent_name="dspy_react",
            turns=turns,
            metadata={"source_file": str(jsonl_path)},
        )
        self.save_trace(trace)
        return trace
