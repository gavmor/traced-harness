"""Agent skill loading, discovery, management, and Agno tool integration."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import frontmatter
import yaml

from traced_harness.telemetry import (
    SKILL_CHARS_LOADED_ATTR,
    SKILL_DISCOVERY_SPAN,
    SKILL_MODE_ATTR,
    SKILL_NAME_ATTR,
    SKILL_PATH_ATTR,
    get_tracer,
    record_skill_activation_span,
)

tracer = get_tracer("traced.harness.skills")



@dataclass
class Skill:
    """Represents an agent skill specification loaded from SKILL.md or directory."""

    name: str
    description: str
    content: str
    path: Path
    metadata: dict[str, Any] = field(default_factory=dict)
    active: bool = False

    @property
    def chars_loaded(self) -> int:
        return len(self.content)


def _skill_markdown(target_path: Path) -> Path:
    """Resolve a path to the markdown file that defines a skill."""
    if target_path.is_file():
        return target_path
    if not target_path.is_dir():
        raise FileNotFoundError(f"Skill file not found: {target_path}")
    named = next(
        (c for c in (target_path / "SKILL.md", target_path / "skill.md") if c.is_file()),
        None,
    )
    if named:
        return named
    # Fallback to any markdown file in the directory.
    md_files = sorted(target_path.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"No SKILL.md found in directory: {target_path}")
    return md_files[0]


def load_skill_file(path: Path | str) -> Skill:
    """Load and parse a SKILL.md file or directory into a Skill object."""
    target_path = _skill_markdown(Path(path).expanduser().resolve())

    try:
        post = frontmatter.loads(target_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML frontmatter in {target_path}: {exc}") from exc
    metadata: dict[str, Any] = dict(post.metadata)
    content = post.content

    # Determine skill name
    if metadata.get("name"):
        name = str(metadata["name"]).strip()
    elif target_path.name.lower() in ("skill.md", "skill"):
        name = target_path.parent.name
    else:
        name = target_path.stem

    description = str(metadata.get("description", "")).strip()

    # Detect bundle subdirectories (scripts, references, examples)
    bundle_dir = target_path.parent
    if bundle_dir.is_dir():
        for bundle_name in ("scripts", "references", "examples"):
            sub = bundle_dir / bundle_name
            if sub.is_dir():
                metadata[f"{bundle_name}_dir"] = sub

    return Skill(
        name=name,
        description=description,
        content=content.strip(),
        path=target_path,
        metadata=metadata,
        active=False,
    )


_NON_SKILL_MARKDOWN = frozenset({"readme.md", "license.md", "contributing.md"})


def _skill_candidates(dir_path: Path) -> Iterator[Path]:
    """Markdown files in a directory that might define a skill, in priority order.

    Subdirectories first (``<skill>/SKILL.md``), then loose markdown files
    (``my-skill.md``) — so a bundled skill wins a name collision with a flat
    one, which is the documented precedence.
    """
    for item in sorted(dir_path.iterdir()):
        if item.is_dir() and not item.name.startswith("."):
            named = next(
                (c for c in (item / "SKILL.md", item / "skill.md") if c.is_file()),
                None,
            )
            if named:
                yield named
    for item in sorted(dir_path.iterdir()):
        if (
            item.is_file()
            and item.suffix.lower() == ".md"
            and item.name.lower() not in _NON_SKILL_MARKDOWN
        ):
            yield item


def scan_skills_dirs(paths: Sequence[Path | str]) -> list[Skill]:
    """Scan directories in priority order, returning skills with unique names."""
    with tracer.start_as_current_span(SKILL_DISCOVERY_SPAN) as span:
        discovered: dict[str, Skill] = {}
        scanned_dirs: list[str] = []

        for p in paths:
            dir_path = Path(p).expanduser().resolve()
            scanned_dirs.append(str(dir_path))
            if not dir_path.is_dir():
                continue
            for candidate in _skill_candidates(dir_path):
                try:
                    skill = load_skill_file(candidate)
                except (OSError, ValueError):
                    continue
                discovered.setdefault(skill.name, skill)

        span.set_attribute("skills.directories", scanned_dirs)
        span.set_attribute("skills.discovered_count", len(discovered))

        return list(discovered.values())


def _search_paths(
    skills_dir: Path | str | None,
    no_skills: bool,
    base_dir: Path | None,
    global_dir: Path | None,
) -> list[Path]:
    """Skill directories in specification priority order, existing ones only."""
    paths: list[Path] = []
    if skills_dir:
        paths.append(Path(skills_dir))
    if no_skills:
        return paths
    base = base_dir or Path.cwd()
    candidates = [
        base / ".skills",
        base / "skills",
        global_dir if global_dir is not None else Path.home() / ".agents" / "skills",
    ]
    paths.extend(c for c in candidates if c.is_dir())
    return paths


def _activate(skill: Skill) -> Skill:
    """Mark a skill active and record the activation span."""
    skill.active = True
    record_skill_activation_span(
        name=skill.name,
        path=skill.path,
        chars_loaded=len(skill.content),
        mode="preload",
    )
    return skill


def _resolve_requested(arg: str, skills_map: dict[str, Skill]) -> Skill:
    """Resolve one ``--skill`` argument: an explicit path, or a discovered name."""
    arg_path = Path(arg).expanduser()
    if arg_path.exists():
        return load_skill_file(arg_path)
    found = next(
        (s for s in skills_map.values() if s.name.lower() == arg.lower()), None
    )
    if found is None:
        raise ValueError(
            f"Skill '{arg}' specified via --skill was not found. "
            f"Available skills: {', '.join(skills_map.keys()) or 'None'}"
        )
    return found


def discover_skills(
    skills_dir: Path | str | None = None,
    skill_args: list[str] | None = None,
    no_skills: bool = False,
    base_dir: Path | None = None,
    global_dir: Path | None = None,
) -> list[Skill]:
    """Discover skills by specification priority, pre-activating requested ones."""
    discovered = scan_skills_dirs(
        _search_paths(skills_dir, no_skills, base_dir, global_dir)
    )
    skills_map: dict[str, Skill] = {s.name: s for s in discovered}

    for arg in skill_args or []:
        skill = _activate(_resolve_requested(arg, skills_map))
        skills_map[skill.name] = skill

    result = list(skills_map.values())
    register_skills(result)
    return result


# Global in-process registry for active/available skills
_skills_registry: dict[str, Skill] = {}


def register_skills(skills: list[Skill]) -> None:
    """Register skills into the global registry for dynamic activation."""
    global _skills_registry
    _skills_registry = {s.name: s for s in skills}


def get_registered_skills() -> dict[str, Skill]:
    """Retrieve all currently registered skills."""
    return _skills_registry


def clear_skills_registry() -> None:
    """Clear all registered skills."""
    global _skills_registry
    _skills_registry = {}


def activate_skill(name: str) -> str:
    """Activate and load the full instructions and guidance for an available skill.

    Args:
        name: The name or identifier of the skill to activate.

    Returns:
        The full instructions of the skill, or an error message if not found.
    """
    skill: Skill | None = None
    for s in _skills_registry.values():
        if s.name.lower() == name.lower():
            skill = s
            break

    if skill is None:
        avail = ", ".join(f"'{k}'" for k in _skills_registry)
        return f"Error: Skill '{name}' not found. Available skills: {avail or 'none'}"

    skill.active = True

    tracer = get_tracer("traced.harness.skills")
    with tracer.start_as_current_span("skill.activate") as span:
        span.set_attribute(SKILL_NAME_ATTR, skill.name)
        span.set_attribute(SKILL_PATH_ATTR, str(skill.path))
        span.set_attribute(SKILL_CHARS_LOADED_ATTR, len(skill.content))
        span.set_attribute(SKILL_MODE_ATTR, "dynamic_tool")

    return (
        f"# Skill Activated: {skill.name}\n\n"
        f"**Description:** {skill.description}\n\n"
        f"## Instructions\n\n{skill.content}"
    )


def build_skill_instructions(skills: list[Skill]) -> str:
    """Construct instructions prompt indexing available skills and inlining active ones."""
    if not skills:
        return ""

    sections: list[str] = []

    active_skills = [s for s in skills if s.active]
    available_skills = [s for s in skills if not s.active]

    if active_skills:
        active_lines = [
            (
                "# Pre-Activated Skills\n"
                "The following skills are currently active in your context. "
                "Follow their instructions and workflows carefully:\n"
            )
        ]
        for s in active_skills:
            active_lines.append(f"## Skill: {s.name}\n{s.content}\n")
        sections.append("\n".join(active_lines).strip())

    if available_skills:
        avail_lines = [
            (
                "# Available Skills (Dynamic Loading)\n"
                "You have access to the following skills. When a user task matches one of these skills, "
                "you MUST call the `activate_skill(name)` tool to retrieve its detailed instructions before proceeding:\n"
            )
        ]
        for s in available_skills:
            avail_lines.append(f"- **{s.name}**: {s.description}")
        sections.append("\n".join(avail_lines).strip())

    return "\n\n".join(sections).strip()


# ---------------------------------------------------------------------------
# External peripheral registration.
#
# A peripheral layered on this harness (a memory plugin, a retrieval service,
# anything) may contribute explicit tools the agent can call and/or implicit
# context hooks that need no call. These are registered into (a) the agent's
# system prompt, via ``build_tool_instructions``, and (b) a dispatch registry
# of active tool names, via ``register_external_tools`` — mirroring the skill
# registry. The harness stores the strings; it does not interpret them.
# ---------------------------------------------------------------------------

_external_tool_registry: dict[str, str] = {}


def register_external_tools(tools: list[str], source: str = "") -> None:
    """Register active tool names -> owning source in the global registry."""
    for tool in tools:
        _external_tool_registry[tool] = source


def get_registered_external_tools() -> dict[str, str]:
    """Return the active tool -> source dispatch registry."""
    return dict(_external_tool_registry)


def clear_external_tools() -> None:
    """Clear the external-tool registry (used between benchmark suites)."""
    _external_tool_registry.clear()


def build_tool_instructions(
    source: str,
    tools: list[str],
    context_hooks: list[str] | None = None,
    system_prompt: str = "",
    heading: str = "Peripheral",
) -> str:
    """Construct the system-prompt section describing an external peripheral,
    its explicit tools, and any implicit context hooks.

    ``heading`` lets the caller name the domain (e.g. "Memory Provider")
    without the harness knowing what that domain is.
    """
    hooks = context_hooks or []
    if not tools and not hooks and not system_prompt:
        return ""
    lines = [f"# {heading}: {source}"]
    if system_prompt:
        lines.append(system_prompt)
    if tools:
        tool_list = ", ".join(f"`{t}`" for t in tools)
        lines.append(f"Available tools: {tool_list}.")
    if hooks:
        hook_list = ", ".join(hooks)
        lines.append(
            f"Context-engine hooks (implicit, no call required): {hook_list}."
        )
    return "\n".join(lines).strip()
