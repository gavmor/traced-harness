# traced-agent

Whole-trace qualitative evaluation, failure taxonomy matrix labeling, and LLM-as-judge compiler for agentic workflows.

---

## The Workflow

Flagging an LLM response as inadequate or wrong requires evaluating the **entire trace**—the complete, ordered record of user queries, intermediate reasoning, tool calls, tool outputs, and final responses—rather than judging the final text output in isolation. Root causes often occur upstream from where the failure becomes visible.

### 1. Open Coding & First-Failure Annotation
- **Identify Upstream Root Cause**: Locate the **first failure point** where the system diverged.
- **Write Freeform Notes**: Record brief open codes capturing the divergence (e.g. *"SQL query dropped pet-friendly filter"*).
- **Apply Binary Judgments**: Holistic **Pass** or **Fail** per turn and session.

### 2. Failure Taxonomy & Matrix Labeling
- **Axial Coding**: Cluster open codes into a structured failure taxonomy (`wrong_tool`, `hallucinated_ui`, `topological_inversion`, `anchoring_drift`, `missing_constraint`).
- **Structured Matrix Labeling**: Multi-label binary indicators (1=present, 0=absent).
- **Prevalence Rates**: Calculate issue frequency ($\sum \text{failures} / N$) to guide system and tool fixes.

### 3. Evaluating Multi-Turn Session Traces
- **Session-Level First**: Judge overall goal attainment before scoring turns.
- **Turn-Level Drill-Down**: Pinpoint the exact step of divergence.
- **Isolate Multi-Turn vs Single-Turn**: Detect multi-turn coherence issues (anchoring, context truncation, instruction drift) vs single-turn capability limitations.

### 4. Trace Review Tooling & LLM-as-Judge Compiler
- Interactive CLI tree renderer displaying the execution graph (User Input → Thoughts → Tool Invocations → Observations → Completion).
- **Judge Compiler**: Compiles annotated failure modes and open-code notes into automated DeepEval `GEval` metrics and pytest test suites.

---

## Quickstart

```bash
# Ingest sortie session traces
traced-agent ingest /path/to/sessions/

# List imported sessions
traced-agent list

# Review execution graphs interactively
traced-agent review

# View failure prevalence matrix
traced-agent matrix

# Compile tagged failure modes into continuous LLM-as-judge tests
traced-agent compile-judges -o tests/evals/test_compiled_evals.py
```
