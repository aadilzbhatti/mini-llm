"""Programs: the unit AlphaEvolve-style search evolves.

A program is the base commit's `mini_llm` source with the contents of its
`# EVOLVE-BLOCK-START <name>` / `# EVOLVE-BLOCK-END <name>` regions replaced, plus
a hyperparameter patch. Children are made by applying SEARCH/REPLACE diffs (the
paper's output format) to a parent:

    <<<<<<< SEARCH
    exact text from the parent
    =======
    replacement
    >>>>>>> REPLACE

`apply_diffs` enforces scope. Every SEARCH must match exactly once, inside one
block. Afterwards the file outside the blocks (marker lines included) must be
byte-identical to the base, so no protected line can change. `static_violations`
then rejects block code that could cheat or escape: referring to `targets`
(labels are in scope in forward()), file/OS/network/pickle access, dynamic
code execution, or torch.load/save.
"""

from __future__ import annotations

import io
import json
import re
import subprocess
import textwrap
import tokenize
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

EDITABLE_FILES = ("mini_llm/model.py", "mini_llm/train.py")  # relative to src/
START = re.compile(r"^[ \t]*# EVOLVE-BLOCK-START (\w+)[ \t]*$")
END = re.compile(r"^[ \t]*# EVOLVE-BLOCK-END (\w+)[ \t]*$")
DIFF = re.compile(r"<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE", re.S)

FORBIDDEN_NAMES = {
    "targets",  # the labels: a block that reads them can leak them into the logits
    "open", "os", "sys", "subprocess", "socket", "shutil", "pathlib", "requests", "urllib", "http",
    "importlib", "builtins", "pickle", "marshal", "ctypes", "eval", "exec", "__import__",
    "globals", "locals", "vars", "breakpoint", "input",
}
FORBIDDEN_ATTRS = {("torch", "load"), ("torch", "save"), ("torch", "hub")}


class ScopeError(ValueError):
    """A diff or program that breaks the evolve-block rules. The message is the reject reason."""


@dataclass
class Program:
    id: str
    parent_id: str | None
    base_commit: str
    blocks: dict[str, str]  # "<file>:<block name>" -> block content (between the marker lines)
    hparams: dict
    rationale: str = ""
    created_by: str = "human"  # human | mutation | <llm model id>
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    diffs: list[dict] = field(default_factory=list)  # [{"search": ..., "replace": ...}] applied to the parent
    # evaluation state (see autolab.evaluate)
    stage: str = "static"
    status: str = "queued"  # queued | running | rejected | contender | accepted | evaluated
    reason: str = ""
    stages: list[dict] = field(default_factory=list)
    runs: dict[str, list[str]] = field(default_factory=dict)  # stage -> run ids
    scores: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Program":
        return cls(**d)


# --- blocks -------------------------------------------------------------------------


def parse_blocks(text: str, path: str = "") -> list[tuple[str, int, int]]:
    """[(name, start_marker_line, end_marker_line)] (0-based), validating pairing and uniqueness."""
    lines = text.split("\n")
    out, open_name, open_at, seen = [], None, None, set()
    for i, line in enumerate(lines):
        if m := START.match(line):
            if open_name is not None:
                raise ScopeError(f"{path}: block {m.group(1)!r} starts inside block {open_name!r}")
            if m.group(1) in seen:
                raise ScopeError(f"{path}: duplicate block {m.group(1)!r}")
            open_name, open_at = m.group(1), i
        elif m := END.match(line):
            if m.group(1) != open_name:
                raise ScopeError(f"{path}: END {m.group(1)!r} doesn't close {open_name!r}")
            out.append((open_name, open_at, i))
            seen.add(open_name)
            open_name = None
    if open_name is not None:
        raise ScopeError(f"{path}: block {open_name!r} never ends")
    return out


def extract_blocks(files: dict[str, str]) -> dict[str, str]:
    blocks = {}
    for path in EDITABLE_FILES:
        lines = files[path].split("\n")
        for name, a, b in parse_blocks(files[path], path):
            blocks[f"{path}:{name}"] = "\n".join(lines[a + 1 : b])
    return blocks


def skeleton(text: str, path: str = "") -> str:
    """The file with every block's content removed: what must never change."""
    lines = text.split("\n")
    keep, cursor = [], 0
    for _, a, b in parse_blocks(text, path):
        keep += lines[cursor : a + 1]
        cursor = b
    keep += lines[cursor:]
    return "\n".join(keep)


def render(base_files: dict[str, str], blocks: dict[str, str]) -> dict[str, str]:
    """Base sources with the program's block contents substituted."""
    out = dict(base_files)
    for path in EDITABLE_FILES:
        text = base_files[path]
        lines = text.split("\n")
        new, cursor = [], 0
        for name, a, b in parse_blocks(text, path):
            new += lines[cursor : a + 1]
            content = blocks.get(f"{path}:{name}")
            new += (content.split("\n") if content else []) if content is not None else lines[a + 1 : b]
            cursor = b
        new += lines[cursor:]
        out[path] = "\n".join(new)
    return out


# --- diffs ---------------------------------------------------------------------------


def parse_diff(text: str) -> list[dict]:
    diffs = [{"search": m.group(1), "replace": m.group(2)} for m in DIFF.finditer(text)]
    if not diffs:
        raise ScopeError("no SEARCH/REPLACE blocks found")
    return diffs


def apply_diffs(base_files: dict[str, str], parent_blocks: dict[str, str], diffs: list[dict]) -> dict[str, str]:
    """Apply SEARCH/REPLACE diffs to the parent's rendered files; return the child's blocks.

    Raises ScopeError (whose message is the reject reason) if a SEARCH doesn't match
    exactly once inside a single block, or if anything outside the blocks changed.
    """
    files = render(base_files, parent_blocks)
    for n, d in enumerate(diffs, 1):
        search, replace = d["search"], d["replace"]
        if not search.strip():
            raise ScopeError(f"diff {n}: empty SEARCH")
        hits = [(p, i) for p in EDITABLE_FILES for i in _find_all(files[p], search)]
        if not hits:
            raise ScopeError(f"diff {n}: SEARCH text not found in model.py/train.py")
        if len(hits) > 1:
            raise ScopeError(f"diff {n}: SEARCH text matches {len(hits)} places; make it unique")
        path, at = hits[0]
        text = files[path]
        if not _inside_one_block(text, at, at + len(search), path):
            raise ScopeError(f"diff {n}: SEARCH text in {path} is not entirely inside one EVOLVE block")
        files[path] = text[:at] + replace + text[at + len(search) :]
    for path in EDITABLE_FILES:
        try:
            ok = skeleton(files[path], path) == skeleton(base_files[path], path)
        except ScopeError as exc:
            raise ScopeError(f"markers broken: {exc}") from None
        if not ok:
            raise ScopeError(f"{path}: text outside the EVOLVE blocks (or a marker line) changed")
    return extract_blocks(files)


def _find_all(text: str, needle: str) -> list[int]:
    out, i = [], text.find(needle)
    while i != -1:
        out.append(i)
        i = text.find(needle, i + 1)
    return out


def _inside_one_block(text: str, start: int, end: int, path: str) -> bool:
    line_starts = [0] + [i + 1 for i, c in enumerate(text) if c == "\n"]

    def line_of(pos: int) -> int:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= pos:
                lo = mid
            else:
                hi = mid - 1
        return lo

    a, b = line_of(start), line_of(max(start, end - 1))
    return any(sa < a and b < sb for _, sa, sb in parse_blocks(text, path))


# --- static rules ----------------------------------------------------------------------


def static_violations(blocks: dict[str, str]) -> list[str]:
    """Forbidden identifiers in block code (comments and strings are ignored)."""
    problems = []
    for key, content in blocks.items():
        try:
            toks = list(tokenize.generate_tokens(io.StringIO(textwrap.dedent(content) + "\n").readline))
        except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
            problems.append(f"{key}: does not tokenize ({exc})")
            continue
        names = [t for t in toks if t.type in (tokenize.NAME, tokenize.OP)]
        for i, t in enumerate(names):
            if t.type != tokenize.NAME:
                continue
            prev = names[i - 1].string if i else ""
            if prev == "." and i >= 2 and (names[i - 2].string, t.string) in FORBIDDEN_ATTRS:
                problems.append(f"{key}: uses {names[i - 2].string}.{t.string}")
            elif prev != "." and t.string in FORBIDDEN_NAMES:
                problems.append(f"{key}: uses forbidden name {t.string!r} (line {t.start[0]})")
            elif prev != "." and t.string == "compile":
                problems.append(f"{key}: uses builtin compile()")
    return problems


# --- hparams ---------------------------------------------------------------------------


def validate_hparams(hparams: dict, spec: dict) -> list[str]:
    """spec: {key: {"min":..,"max":..} | {"choices": [...]}, ...}; unknown keys are errors."""
    problems = []
    for k, v in hparams.items():
        rule = spec.get(k)
        if rule is None:
            problems.append(f"hparam {k!r} is not evolvable")
        elif "choices" in rule and v not in rule["choices"]:
            problems.append(f"hparam {k}={v!r} not in {rule['choices']}")
        elif "min" in rule and not (isinstance(v, (int, float)) and rule["min"] <= v <= rule["max"]):
            problems.append(f"hparam {k}={v!r} outside [{rule['min']}, {rule['max']}]")
        elif rule.get("int") and isinstance(v, float) and v != int(v):
            problems.append(f"hparam {k}={v!r} must be an integer")
    if {"n_embd", "n_head"} <= hparams.keys() and hparams["n_embd"] % hparams["n_head"]:
        problems.append(f"n_embd {hparams['n_embd']} not divisible by n_head {hparams['n_head']}")
    if "min_lr" in hparams and "lr" in hparams and hparams["min_lr"] > hparams["lr"]:
        problems.append("min_lr > lr")
    return problems


# --- base sources ----------------------------------------------------------------------


def base_sources(repo: Path, commit: str) -> dict[str, str]:
    """Every file under src/mini_llm at `commit`, keyed relative to src/ (e.g. mini_llm/model.py)."""
    names = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", commit, "src/mini_llm"],
                           capture_output=True, text=True, check=True).stdout.split()
    out = {}
    for name in names:
        blob = subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{name}"], capture_output=True, check=True)
        try:
            out[name.removeprefix("src/")] = blob.stdout.decode()
        except UnicodeDecodeError:
            continue  # binary assets aren't needed to train
    return out


def materialize(files: dict[str, str], dest: Path) -> Path:
    """Write a program's sources to dest/src and return dest/src (for PYTHONPATH)."""
    src = dest / "src"
    for rel, text in files.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_text(text)
    return src


def save(program: Program, programs_dir: Path) -> Path:
    programs_dir.mkdir(parents=True, exist_ok=True)
    path = programs_dir / f"{program.id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(program.to_dict(), indent=2))
    tmp.replace(path)
    return path


def load(path: Path) -> Program:
    return Program.from_dict(json.loads(Path(path).read_text()))
