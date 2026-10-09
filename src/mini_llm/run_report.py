"""One shareable report per training run: everything needed to judge it, in one file.

    uv run mini-llm-run-report <run_id>          # -> reports/<eval report name>.md, .html, .pdf
    uv run mini-llm-run-report --all             # every run that has finished its evals

Written automatically when the Mac's auto-eval finishes a run (mini_llm.auto_eval). Meant to be handed
to a person or another model for evaluation without copying anything by hand:

  .md   every number as text: run facts, training curves as tables, the eval suite, a comparison with
        every other evaluated model at the same context, all 100 generation samples, and a glossary of
        what each metric means. The format to give another model (text survives; PDF tables often don't).
  .pdf  the same content plus the loss plot, printed from .html by headless Chrome (skipped, with a
        note, when Chrome isn't installed).
  .html the same, self-contained (plot embedded), what the PDF is printed from.

Sources, all already on disk: runs/<id>.status.json (config, cost, full_val curve), runs/<id>.log (the
quick eval curve), evals/<report>.json (the eval suite and samples), the loss plot from baselines.json,
and the dataset's MANIFEST.md / mix.json.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from mini_llm.remote import costs

REPORTS_DIR = "reports"
CHROME = os.environ.get("MINI_LLM_CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
MAX_CURVE_ROWS = 24


# --- blocks: rendered to Markdown and to HTML from the same content ------------------


@dataclass
class H:
    level: int
    text: str


@dataclass
class P:
    text: str


@dataclass
class Table:
    head: list[str]
    rows: list[list[str]]


@dataclass
class Pre:
    text: str


@dataclass
class Img:
    path: Path
    alt: str


def to_markdown(blocks: list) -> str:
    out = []
    for b in blocks:
        if isinstance(b, H):
            out.append(f"{'#' * b.level} {b.text}")
        elif isinstance(b, P):
            out.append(b.text)
        elif isinstance(b, Table):
            cell = lambda s: str(s).replace("|", "\\|").replace("\n", " ")
            out.append(
                "\n".join(
                    ["| " + " | ".join(map(cell, b.head)) + " |", "|" + "---|" * len(b.head)]
                    + ["| " + " | ".join(map(cell, r)) + " |" for r in b.rows]
                )
            )
        elif isinstance(b, Pre):
            fence = "````" if "```" in b.text else "```"
            out.append(f"{fence}text\n{b.text}\n{fence}")
        elif isinstance(b, Img):
            out.append(f"*[{b.alt}: see the PDF/HTML version of this report, or `{b.path}`]*")
    return "\n\n".join(out) + "\n"


def _inline(text: str) -> str:
    """The little Markdown the report's own text uses: **bold** and `code`."""
    t = html.escape(text)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    return re.sub(r"`(.+?)`", r"<code>\1</code>", t)


def to_html(blocks: list, title: str) -> str:
    body = []
    for b in blocks:
        if isinstance(b, H):
            body.append(f"<h{b.level}>{_inline(b.text)}</h{b.level}>")
        elif isinstance(b, P):
            body.append(f"<p>{_inline(b.text)}</p>")
        elif isinstance(b, Table):
            head = "".join(f"<th>{_inline(str(c))}</th>" for c in b.head)
            rows = "".join("<tr>" + "".join(f"<td>{_inline(str(c))}</td>" for c in r) + "</tr>" for r in b.rows)
            body.append(f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>")
        elif isinstance(b, Pre):
            body.append(f"<pre>{html.escape(b.text)}</pre>")
        elif isinstance(b, Img) and b.path.exists():
            data = base64.b64encode(b.path.read_bytes()).decode()
            body.append(f'<img alt="{html.escape(b.alt)}" src="data:image/png;base64,{data}">')
    css = (
        "body{font:13px/1.45 -apple-system,Helvetica,Arial,sans-serif;color:#111;margin:28px;max-width:1000px}"
        "h1{font-size:22px}h2{font-size:17px;margin-top:26px;border-bottom:1px solid #ccc}h3{font-size:14px}"
        "table{border-collapse:collapse;margin:8px 0;font-variant-numeric:tabular-nums}"
        "th,td{border:1px solid #ccc;padding:3px 7px;text-align:left;vertical-align:top}th{background:#f2f2f2}"
        "pre{white-space:pre-wrap;background:#f7f7f7;border:1px solid #e3e3e3;padding:8px;font-size:11.5px}"
        "code{font-size:12px;background:#f2f2f2;padding:0 3px}img{max-width:100%}"
        "@media print{h2{break-after:avoid}table,pre,img{break-inside:avoid}}"
    )
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{css}</style></head><body>{''.join(body)}</body></html>\n"
    )


def print_pdf(html_path: Path, pdf_path: Path, chrome: str | None = None, timeout: float = 120) -> str | None:
    """Print the HTML to PDF with headless Chrome. Returns None, or why no PDF was written.

    Headless Chrome often writes the PDF and then doesn't exit, so this waits for the file to appear and
    stop growing, then ends Chrome itself, rather than waiting on the process."""
    chrome = chrome or CHROME  # read at call time, not import time
    if not Path(chrome).exists():
        return f"no PDF: Chrome not found at {chrome} (set MINI_LLM_CHROME)"
    pdf_path.unlink(missing_ok=True)
    profile = pdf_path.parent / f".chrome-{os.getpid()}"
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-pdf-header-footer",
        f"--user-data-dir={profile}",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    deadline, last_size, stable_since = time.time() + timeout, -1, None
    try:
        while time.time() < deadline and proc.poll() is None:
            size = pdf_path.stat().st_size if pdf_path.exists() else -1
            if size > 0 and size == last_size:
                stable_since = stable_since or time.time()
                if time.time() - stable_since >= 1.5:
                    break
            else:
                stable_since = None
            last_size = size
            time.sleep(0.25)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, 15)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, 9)
        subprocess.run(["rm", "-rf", str(profile)], check=False)
    if pdf_path.exists() and pdf_path.stat().st_size > 0:
        return None
    return "no PDF: Chrome timed out" if time.time() >= deadline else "no PDF: Chrome wrote nothing"


# --- content -----------------------------------------------------------------------


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def f4(v) -> str:
    return "–" if v is None else f"{v:.4f}"


def pct(v) -> str:
    return "–" if v is None else f"{100 * v:.0f}%"


def _eval_curve(log: Path) -> list[tuple[int, float, float]]:
    rows = []
    pat = re.compile(r"^step\s+(\d+) \| eval_train_loss ([\d.]+) \| eval_val_loss ([\d.]+)")
    try:
        for line in log.read_text(errors="replace").splitlines():
            if m := pat.match(line):
                rows.append((int(m[1]), float(m[2]), float(m[3])))
    except OSError:
        pass
    seen, out = set(), []
    for r in rows:  # DDP / resumes can log a step twice
        if r[0] not in seen:
            seen.add(r[0]), out.append(r)
    return out


def _thin(rows: list, n: int = MAX_CURVE_ROWS) -> list:
    if len(rows) <= n:
        return rows
    step = (len(rows) - 1) / (n - 1)
    return [rows[round(i * step)] for i in range(n)]


def _plot_for(repo: Path, checkpoint_stem: str) -> Path | None:
    for row in _read_json(repo / "baselines.json") or []:
        if Path(row.get("checkpoint") or "").stem == checkpoint_stem and row.get("plot"):
            p = repo / row["plot"]
            return p if p.exists() else None
    return None


def _dataset_dir(repo: Path, args: dict) -> Path | None:
    """data/<name>/ for the run's --tokens (a repo path, or /data/<name>/... as Modal saw it)."""
    tokens = args.get("tokens")
    if not isinstance(tokens, str):
        return None
    rel = "data/" + tokens.removeprefix("/data/") if tokens.startswith("/data/") else tokens
    d = (repo / rel).parent
    return d if d.is_dir() else None


GLOSSARY = [
    (
        "full_val@c",
        "Mean next-token loss (nats) over every token of the fixed 918,728-token validation set, in windows of c tokens. Lower is better. full_val@1024 is the headline number.",
    ),
    (
        "L(c)",
        "Fixed-target context curve: loss on the same 8,000 target tokens given exactly c tokens of history. Comparable across models at equal c.",
    ),
    (
        "cb@W",
        "Context benefit: loss with the real preceding W/2 tokens vs. a prefix from a different document, on identical windows. Nats of help from relevant context (± standard error).",
    ),
    (
        "retrieval@d",
        "Forced choice among 10 candidates (chance 10%): can the model pick the token that appeared d tokens earlier? Tests using information far back in context.",
    ),
    ("rep4", "Share of repeated 4-grams in a generated sample (0 = no repetition). Median over 100 samples."),
    ("loops", "Samples (of 100) that fall into a verbatim repetition loop, with its 95% CI."),
    ("topic held", "Share of samples still mentioning the prompt's subject near the end of the 256-token generation."),
    ("distinct-2 / -4", "Distinct bigrams / 4-grams over total, a diversity measure."),
    (
        "generation protocol",
        "20 frozen prompts x 5 seeds, 256 new tokens, temperature 0.7, top-k 40, stop at EOS; identical for every model.",
    ),
]


def _comparison(repo: Path, me: dict) -> Table:
    from mini_llm.evals import eval_results
    from mini_llm.samples import label

    T = me["config"]["block_size"]
    rows = []
    for r in eval_results(repo / "evals"):
        if r["config"]["block_size"] != T:
            continue
        q, s = r["quality"], (r.get("samples") or {}).get("summary") or {}
        cc = (r.get("context_curve") or {}).get("by_context") or {}
        ret = (r.get("retrieval") or {}).get("by_distance") or {}
        acc = lambda d: (ret.get(str(d)) or {}).get("accuracy")
        mine = Path(r["checkpoint"]).stem == Path(me["checkpoint"]).stem
        rows.append(
            (
                q.get(f"full_val@{T}", 9e9),
                [
                    ("**this run** · " if mine else "") + label(r),
                    f4(q.get(f"full_val@{T}")),
                    f4((cc.get(str(T)) or {}).get("loss")),
                    "–" if s.get("rep4") is None else f"{s['rep4']:.3f}",
                    "–" if s.get("looped") is None else f"{s['looped']}/{s.get('n', 100)}",
                    pct(s.get("topic")),
                    pct(acc(256)),
                    pct(acc(496)),
                ],
            )
        )
    rows.sort(key=lambda x: x[0])
    return Table(
        [
            "model",
            f"full_val@{T}",
            f"L(c={T})",
            "rep4 median",
            "loops",
            "topic held",
            "retrieval@256",
            "retrieval@496",
        ],
        [r for _, r in rows],
    )


def build(repo: Path, run_id: str) -> tuple[list, str, dict]:
    """(blocks, file stem, facts) for one finished, evaluated run."""
    status = _read_json(repo / "runs" / f"{run_id}.status.json")
    if not status:
        raise FileNotFoundError(f"no run {run_id}")
    ev = status.get("eval") or {}
    if ev.get("state") != "done":
        raise ValueError(f"{run_id}: evals not done (state {ev.get('state')!r})")
    stem = ev["report"]
    r = _read_json(repo / "evals" / f"{stem}.json")
    if not r:
        raise FileNotFoundError(f"evals/{stem}.json")
    args, cfg, met = status.get("args") or {}, r["config"], status.get("metrics") or {}
    remote = status.get("remote") or {}
    T = cfg["block_size"]
    sysm, q, smp = r.get("training_systems") or {}, r["quality"], r.get("samples") or {}
    s = smp.get("summary") or {}
    cost = costs.run_cost(status, None, costs.load_rates(repo / "runs"), time.time())
    batch, steps = int(args.get("batch-size", 0) or 0), int(r.get("step") or 0)
    data_dir = _dataset_dir(repo, args)
    tokens_seen = sysm.get("tokens") or (batch * T * steps if batch else None)
    train_tokens = None
    if data_dir and (mix := _read_json(data_dir / "mix.json")):
        train_tokens = mix.get("train_tokens")
    elif data_dir and (data_dir / "MANIFEST.md").exists():
        m = re.search(r"\| train\.pt \| ([\d,]+)", (data_dir / "MANIFEST.md").read_text())
        train_tokens = int(m[1].replace(",", "")) if m else None

    facts = {
        "run": status.get("name") or run_id,
        "full_val": q.get(f"full_val@{T}"),
    }
    B: list = [
        H(1, f"Training run report: {status.get('name') or run_id}"),
        P(
            f"Run `{run_id}` · checkpoint `{Path(r['checkpoint']).name}` · generated "
            f"{time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. "
            "A decoder-only Transformer language model (GPT-2 tokenizer, 50,257 tokens) trained from scratch; "
            "every number below is from the project's fixed evaluation harness, identical across models. "
            "Lower loss is better; the glossary at the end defines each metric."
        ),
        H(2, "Headline"),
        Table(
            ["metric", "value"],
            [
                [f"full_val@{T} (nats)", f4(q.get(f"full_val@{T}"))],
                [
                    f"fixed-target L(c={T})",
                    f4(((r.get("context_curve") or {}).get("by_context") or {}).get(str(T), {}).get("loss")),
                ],
                ["rep4 median (generation)", "–" if s.get("rep4") is None else f"{s['rep4']:.3f}"],
                [
                    "loops",
                    "–" if s.get("looped") is None else f"{s['looped']}/{s.get('n')} (95% CI {s.get('loop_ci95')})",
                ],
                ["topic held", pct(s.get("topic"))],
                ["final eval train / val loss", f"{f4(met.get('eval_train_loss'))} / {f4(met.get('eval_val_loss'))}"],
                ["cost", "–" if not cost or cost.get("usd") is None else f"${cost['usd']:.2f} ({cost['source']})"],
            ],
        ),
        H(2, "Run"),
        Table(
            ["", ""],
            [
                [
                    "model",
                    f"d_model {cfg['n_embd']} · {cfg['n_layer']} layers · {cfg['n_head']} heads · context {T} · {r['params']:,} params",
                ],
                [
                    "positions / attention",
                    f"{'RoPE' if cfg.get('use_rope_embeddings') else 'learned absolute'} · {'fused (SDPA)' if cfg.get('fused_attention') else 'per-head'} attention · tied embeddings · dropout {cfg.get('dropout')}",
                ],
                ["data", f"`{args.get('tokens')}`" + (f" ({train_tokens:,} train tokens)" if train_tokens else "")],
                ["validation", f"`{args.get('val-tokens')}` (eval suite: `{r.get('val_tokens')}`)"],
                [
                    "schedule",
                    f"{steps:,} steps · batch {batch} x {T} = {batch * T:,} tokens/step · lr {args.get('lr')} cosine to {args.get('min-lr')}"
                    + (
                        f" · restart-lr {args['restart-lr']} from `{args.get('resume')}`"
                        if args.get("restart-lr")
                        else ""
                    )
                    + (f" · warmup {args['warmup-tokens']:,} tokens" if args.get("warmup-tokens") else ""),
                ],
                [
                    "tokens seen",
                    (
                        f"{tokens_seen:,}"
                        + (f" (~{tokens_seen / train_tokens:.2f} passes)" if tokens_seen and train_tokens else "")
                        if tokens_seen
                        else "–"
                    ),
                ],
                ["seed", str(args.get("seed", "–"))],
                [
                    "hardware",
                    f"{sysm.get('device', '–')} x{sysm.get('world_size', '–')} · {sysm.get('train_tokens_per_sec', 0):,.0f} tokens/s · peak {sysm.get('peak_mem_gb', '–')} GB",
                ],
                [
                    "wall time",
                    f"{(status.get('duration_sec') or 0) / 3600:.2f} h ({status.get('started')} → {status.get('finished')})",
                ],
                [
                    "code",
                    f"git {(remote.get('git_sha') or '')[:7] or '–'}"
                    + (f" · Modal {remote.get('gpus')} · app {remote.get('app_id')}" if remote else ""),
                ],
            ],
        ),
        H(2, "Training curves"),
    ]
    if plot := _plot_for(repo, stem):
        B.append(Img(plot, "Loss plot"))
    fv = met.get("full_val_curve") or []
    if fv:  # four step/value pairs per row: the whole curve, compactly
        cells = [(f"{a:,}", f4(b)) for a, b in fv]
        rows = [sum(cells[i : i + 4], ()) for i in range(0, len(cells), 4)]
        B += [
            H(3, "Full validation loss (all val tokens)"),
            Table(["step", "full_val"] * 4, [list(r) + ["", ""] * (4 - len(r) // 2) for r in rows]),
        ]
    ec = _eval_curve(repo / "runs" / f"{run_id}.log")
    if ec:
        B += [
            H(3, f"Quick eval loss ({len(ec)} points, shown {min(len(ec), MAX_CURVE_ROWS)})"),
            Table(
                ["step", "eval train", "eval val", "gap"],
                [[f"{a:,}", f4(b), f4(c), f"{b - c:+.4f}"] for a, b, c in _thin(ec)],
            ),
        ]

    B += [H(2, "Evaluation suite"), H(3, "Loss by window size and by position in the window")]
    windows = sorted(int(k.split("@")[1]) for k in q if k.startswith("full_val@"))
    pos_keys = list((q.get(f"by_position@{T}") or {}).keys())
    B.append(
        Table(
            ["window", "full_val"] + [f"pos {k}" for k in pos_keys],
            [
                [str(w), f4(q[f"full_val@{w}"])] + [f4((q.get(f"by_position@{w}") or {}).get(k)) for k in pos_keys]
                for w in windows
            ],
        )
    )
    cc = r.get("context_curve") or {}
    if by := cc.get("by_context"):
        B += [
            H(3, "Fixed-target context curve L(c)"),
            P(cc.get("protocol", "")),
            Table(
                ["history c", "loss"] + (["gain from doubling"] if cc.get("gain") else []),
                [
                    [c, f4(v.get("loss"))]
                    + (
                        [f4((cc.get("gain") or {}).get(f"{int(c) // 2}->{c}", {}).get("nats"))]
                        if cc.get("gain")
                        else []
                    )
                    for c, v in by.items()
                ],
            ),
        ]
    if cb := r.get("context_benefit"):
        B += [
            H(3, "Context benefit"),
            Table(
                ["window", "real prefix", "other-doc prefix", "benefit (nats ± SE)", "windows"],
                [
                    [
                        k,
                        f4(v["loss_real_prefix"]),
                        f4(v["loss_other_doc_prefix"]),
                        f"{v['benefit_nats']:.4f} ± {v['benefit_se']:.4f}",
                        str(v["windows"]),
                    ]
                    for k, v in cb.items()
                ],
            ),
        ]
    if ret := r.get("retrieval"):
        B += [
            H(
                3,
                f"Retrieval ({ret.get('candidates')} candidates, chance {pct(ret.get('chance'))}, {ret.get('trials_per_distance')} trials each)",
            ),
            Table(
                ["distance"] + list(ret["by_distance"]),
                [["accuracy"] + [pct(v.get("accuracy")) for v in ret["by_distance"].values()]],
            ),
        ]
    if s:
        B += [
            H(3, "Generation (summary over 100 samples)"),
            P(smp.get("protocol", "")),
            Table(
                [
                    "rep4 median / p90",
                    "loops (95% CI)",
                    "first loop at (median)",
                    "distinct-2 / -4",
                    "topic held",
                    "topic span (median)",
                    "EOS",
                    "mean tokens",
                ],
                [
                    [
                        f"{s.get('rep4')} / {s.get('rep4_p90')}",
                        f"{s.get('looped')}/{s.get('n')} {s.get('loop_ci95')}",
                        str(s.get("loop_onset_median")),
                        f"{s.get('distinct2')} / {s.get('distinct4')}",
                        pct(s.get("topic")),
                        str(s.get("topic_span_median")),
                        str(s.get("eos")),
                        str(s.get("tokens")),
                    ]
                ],
            ),
        ]
    if inf := r.get("inference"):
        B += [
            H(3, "Inference (Mac, single sequence)"),
            P(
                f"prefill at full context {inf.get('prefill_ms_full_context')} ms · decode {inf.get('decode_tokens_per_sec')} tokens/s · {inf.get('memory_gb')} GB · {inf.get('note', '')}"
            ),
        ]

    B += [
        H(2, f"Compared with every evaluated model at context {T}"),
        P("Sorted by full_val; same harness, validation set, prompts and seeds for every row."),
        _comparison(repo, r),
    ]

    if data_dir and (data_dir / "MANIFEST.md").exists():
        B += [H(2, f"Dataset: {data_dir.name}"), Pre((data_dir / "MANIFEST.md").read_text().strip())]

    B += [H(2, "Glossary"), Table(["metric", "meaning"], [list(g) for g in GLOSSARY])]

    if prompts := smp.get("prompts"):
        B.append(H(2, "Appendix: all generation samples"))
        for p in prompts:
            B.append(H(3, f"{p['label']}: \u201c{p['prompt']}\u201d"))
            for j, d in enumerate(p["draws"]):
                tags = [
                    f"rep4 {d.get('rep4')}",
                    "loops" if d.get("looped") else "no loop",
                    "topic held" if d.get("topic") else "topic lost",
                ]
                if d.get("eos"):
                    tags.append("ended at EOS")
                B += [P(f"**draw {j + 1}** · " + " · ".join(tags)), Pre(p["prompt"] + d.get("text", ""))]
    return B, stem, facts


def write(repo: Path, run_id: str, pdf: bool = True) -> dict[str, Path | str]:
    blocks, stem, _ = build(repo, run_id)
    out = repo / REPORTS_DIR
    out.mkdir(exist_ok=True)
    title = f"Training run report: {stem}"
    md, page = out / f"{stem}.md", out / f"{stem}.html"
    md.write_text(to_markdown(blocks))
    page.write_text(to_html(blocks, title))
    written: dict[str, Path | str] = {"md": md, "html": page}
    if pdf:
        why = print_pdf(page, out / f"{stem}.pdf")
        written["pdf"] = why or out / f"{stem}.pdf"
    return written


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Write a shareable report (md, html, pdf) for evaluated training runs.")
    p.add_argument("run_ids", nargs="*")
    p.add_argument("--all", action="store_true", help="Every run whose evals are done.")
    p.add_argument("--repo", type=Path, default=Path("."))
    p.add_argument("--no-pdf", action="store_true")
    args = p.parse_args(argv)
    repo = args.repo.resolve()
    ids = list(args.run_ids)
    if args.all:
        for path in sorted((repo / "runs").glob("*.status.json")):
            if ((_read_json(path) or {}).get("eval") or {}).get("state") == "done":
                ids.append(path.name.removesuffix(".status.json"))
    for run_id in ids:
        try:
            for kind, where in write(repo, run_id, pdf=not args.no_pdf).items():
                print(f"{run_id}: {kind} -> {where}")
        except (FileNotFoundError, ValueError) as exc:
            print(f"{run_id}: skipped ({exc})")


if __name__ == "__main__":
    main()
