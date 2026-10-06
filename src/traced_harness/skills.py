"""Agent skill loading, discovery, management, and Agno tool integration."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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

FRONTMATTER_PATTERN = re.compile(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", re.DOTALL)


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


def load_skill_file(path: Path | str) -> Skill:
    """Load and parse a SKILL.md file or directory into a Skill object."""
    target_path = Path(path).expanduser().resolve()

    if target_path.is_dir():
        candidates = [
            target_path / "SKILL.md",
            target_path / "skill.md",
        ]
        found_file = next((c for c in candidates if c.is_file()), None)
        if not found_file:
            # Fallback to any markdown file in directory
            md_files = list(target_path.glob("*.md"))
            if md_files:
                found_file = md_files[0]
            else:
                raise FileNotFoundError(
                    f"No SKILL.md found in directory: {target_path}"
                )
        target_path = found_file
    elif not target_path.is_file():
        raise FileNotFoundError(f"Skill file not found: {target_path}")

    raw_text = target_path.read_text(encoding="utf-8")
    metadata: dict[str, Any] = {}
    content = raw_text

    match = FRONTMATTER_PATTERN.match(raw_text)
    if match:
        yaml_content = match.group(1)
        content = match.group(2)
        try:
            parsed = yaml.safe_load(yaml_content)
            if isinstance(parsed, dict):
                metadata = parsed
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML frontmatter in {target_path}: {exc}"
            ) from exc

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


def scan_skills_dirs(paths: list[Path | str]) -> list[Skill]:
    """Scan directories in priority order and return discovered skills without duplicate names."""
    with tracer.start_as_current_span(SKILL_DISCOVERY_SPAN) as span:
        discovered: dict[str, Skill] = {}
        scanned_dirs: list[str] = []

        for p in paths:
            dir_path = Path(p).expanduser().resolve()
            scanned_dirs.append(str(dir_path))
            if not dir_path.is_dir():
                continue

            # 1. Scan subdirectories containing SKILL.md or skill.md
            for item in sorted(dir_path.iterdir()):
                if item.is_dir() and not item.name.startswith("."):
                    for candidate in (item / "SKILL.md", item / "skill.md"):
                        if candidate.is_file():
                            try:
                                skill = load_skill_file(candidate)
                                if skill.name not in discovered:
                                    discovered[skill.name] = skill
                            except (OSError, ValueError):
                                pass
                            break

            # 2. Scan direct markdown files (e.g. my-skill.md)
            for item in sorted(dir_path.iterdir()):
                if (
                    item.is_file()
                    and item.suffix.lower() == ".md"
                    and item.name.lower()
                    not in ("readme.md", "license.md", "contributing.md")
                ):
                    try:
                        skill = load_skill_file(item)
                        if skill.name not in discovered:
                            discovered[skill.name] = skill
                    except (OSError, ValueError):
                        pass

        span.set_attribute("skills.directories", scanned_dirs)
        span.set_attribute("skills.discovered_count", len(discovered))

        return list(discovered.values())


def discover_skills(
    skills_dir: Path | str | None = None,
    skill_args: list[str] | None = None,
    no_skills: bool = False,
    base_dir: Path | None = None,
    global_dir: Path | None = None,
) -> list[Skill]:
    """Discover skills according to specification priority and pre-activate any requested skills."""
    search_paths: list[Path] = []

    # 1. Explicit CLI skills directory
    if skills_dir:
        search_paths.append(Path(skills_dir))

    # 2. Project-local and global directories (unless disabled)
    if not no_skills:
        base = base_dir or Path.cwd()
        local_hidden = base / ".skills"
        if local_hidden.is_dir():
            search_paths.append(local_hidden)
        local_skills = base / "skills"
        if local_skills.is_dir():
            search_paths.append(local_skills)

        target_global = (
            global_dir
            if global_dir is not None
            else (Path.home() / ".agents" / "skills")
        )
        if target_global.is_dir():
            search_paths.append(target_global)

    discovered = scan_skills_dirs(search_paths)
    skills_map: dict[str, Skill] = {s.name: s for s in discovered}

    # 3. Handle explicit --skill <path|name> arguments
    if skill_args:
        for arg in skill_args:
            arg_path = Path(arg).expanduser()
            if arg_path.exists():
                # Explicit path
                explicit_skill = load_skill_file(arg_path)
                explicit_skill.active = True
                skills_map[explicit_skill.name] = explicit_skill
                record_skill_activation_span(
                    name=explicit_skill.name,
                    path=explicit_skill.path,
                    chars_loaded=len(explicit_skill.content),
                    mode="preload",
                )
            else:
                # Name lookup in discovered skills
                target: Skill | None = None
                for s in skills_map.values():
                    if s.name.lower() == arg.lower():
                        target = s
                        break
                if target is not None:
                    target.active = True
                    record_skill_activation_span(
                        name=target.name,
                        path=target.path,
                        chars_loaded=len(target.content),
                        mode="preload",
                    )
                else:
                    raise ValueError(
                        f"Skill '{arg}' specified via --skill was not found. "
                        f"Available skills: {', '.join(skills_map.keys()) or 'None'}"
                    )

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
# Memory-provider registration.
#
# A memory provider (see ``plugins.MemoryPluginAdapter``) contributes recall
# tools (``cashew_query``, ``recall``, ...) and/or implicit context-engine
# hooks (``nachos`` manifest/prefetch, ``chronicle`` compression). These are
# registered into (a) the agent's system prompt, via
# ``build_memory_instructions``, and (b) a dispatch registry of active memory
# tool names, via ``register_memory_tools`` — mirroring the skill registry.
# ---------------------------------------------------------------------------

_memory_tool_registry: dict[str, str] = {}


def register_memory_tools(tools: list[str], provider: str = "") -> None:
    """Register active memory tool names -> provider in the global registry."""
    for tool in tools:
        _memory_tool_registry[tool] = provider


def get_registered_memory_tools() -> dict[str, str]:
    """Return the active memory-tool -> provider dispatch registry."""
    return dict(_memory_tool_registry)


def clear_memory_tools() -> None:
    """Clear the memory-tool registry (used between benchmark suites)."""
    _memory_tool_registry.clear()


def build_memory_instructions(
    provider: str,
    tools: list[str],
    context_hooks: list[str] | None = None,
    system_prompt: str = "",
) -> str:
    """Construct the system-prompt section describing the active memory
    provider, its explicit recall tools, and any implicit context hooks.
    """
    hooks = context_hooks or []
    if not tools and not hooks and not system_prompt:
        return ""
    lines = [f"# Memory Provider: {provider}"]
    if system_prompt:
        lines.append(system_prompt)
    if tools:
        tool_list = ", ".join(f"`{t}`" for t in tools)
        lines.append(f"Available memory tools: {tool_list}.")
    if hooks:
        hook_list = ", ".join(hooks)
        lines.append(
            f"Context-engine hooks (implicit, no call required): {hook_list}."
        )
    return "\n".join(lines).strip()
