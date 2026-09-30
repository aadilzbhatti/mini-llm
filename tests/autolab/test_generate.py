"""M4: program database, prompt sampler, proposer failure modes and the mutation fallback."""

import json
import os
import random
import subprocess

import pytest

from autolab import evaluate as ev
from autolab.config import REPO_ROOT
from autolab.database import DBConfig, elites, maybe_migrate, next_island, sample
from autolab.generate import generate_one, mutate
from autolab.llm import LLMError, call
from autolab.program import Program, save
from autolab.prompt import REPLY_SCHEMA, build_prompt

HP = {"n_embd": 256, "n_head": 4, "n_layer": 4, "dropout": 0.0, "batch_size": 64, "lr": 1.2e-3,
      "min_lr": 2e-6, "warmup_steps": 100, "weight_decay": 0.0}


def report(loss, params=16_105_297, tps=41_000.0):
    return {"summary": {"final_full_val_loss": loss, "final_train_loss_smooth": loss - 0.1, "final_val_loss_smooth": loss,
                        "gap": 0.1, "train_slope_tail": {"rel_change": -0.001}, "val_slope_tail": {"rel_change": -0.001},
                        "gap_trend": {"change": 0.01}, "lr_peak": 1.2e-3, "lr_final": 2e-6},
            "scale": {"tokens_seen": 81_920_000, "epochs": 3.99, "params": params, "non_embedding_params": 3_200_000,
                      "tokens_per_param": 5.1, "dataset_tokens": 20_543_855, "val_tokens": 918_728},
            "health": {"nan_or_inf": False, "spikes": {"count": 1}, "grad_norm": {"median": 0.5, "p95": 0.55, "max_over_median": 1.2}},
            "performance": {"tokens_per_sec": tps, "wall_s": 2100.0, "train_wall_s": 2000.0}}


@pytest.fixture
def lab(tmp_path):
    runs = tmp_path / "runs"
    for i, (screen, full) in enumerate([(5.95, 4.72), (5.96, 4.71), (5.94, 4.73)], 1):
        for kind, loss in (("screen", screen), ("full", full)):
            (runs / f"{kind}{i}").mkdir(parents=True)
            (runs / f"{kind}{i}" / "report.json").write_text(json.dumps(report(loss)))
    (runs / "full1" / "diagnosis.json").write_text(json.dumps({"primary": "data_limited", "notes": ["n1"], "labels": [
        {"name": "data_limited", "confidence": 0.75, "evidence": {"epochs": 3.99}, "suggestion": "more data"}]}))
    paths = ev.Paths(tmp_path / "state" / "evolve" / "s")
    head = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    ev.init_session(head, HP, {"screen": ["screen1", "screen2", "screen3"], "full": ["full1", "full2", "full3"]},
                    paths=paths, runs_dir=runs, name="s",
                    budgets={"screen_tokens": 10_485_760, "full_tokens": 81_920_000, "eval": {"eval_interval": 100}})
    return {"paths": paths, "runs": runs, "tmp": tmp_path}


def add(lab, pid, status, full_mean, island, params=16_000_000, tps=41_000.0, parent="p0"):
    p = Program(id=pid, parent_id=parent, base_commit="x", blocks={}, hparams=HP, status=status, stage="done",
                scores={"full_mean": full_mean, "params": params, "tokens_per_sec": tps}, island=island,
                rationale=f"idea {pid}")
    save(p, lab["paths"].programs)
    return p


# --- database ------------------------------------------------------------------------------


def test_sample_exploit_and_inspirations(lab):
    add(lab, "p1", "evaluated", 4.70, 0)
    add(lab, "p2", "contender", 4.69, 0, tps=30_000.0)  # another cell
    add(lab, "p3", "rejected", None, 0)
    add(lab, "p4", "evaluated", 4.68, 1)                 # other island: not a parent on island 0
    progs, session = ev.programs(lab["paths"]), ev.load_session(lab["paths"])
    cfg = DBConfig(p_exploit=1.0)
    parent, insp = sample(progs, session, 0, cfg, random.Random(0))
    assert parent.id == "p2"
    ids = [q.id for q in insp]
    assert ids[:2] == ["p4", "p1"] and "p2" not in ids and "p3" not in ids
    grid = elites(list(progs.values()), cfg)
    assert {p.id for p in grid.values()} >= {"p2", "p4"}


def test_sample_explore_uses_random_cells(lab):
    add(lab, "p1", "evaluated", 4.70, 0, params=10_000_000)
    add(lab, "p2", "evaluated", 4.65, 0, params=22_000_000)
    progs, session = ev.programs(lab["paths"]), ev.load_session(lab["paths"])
    seen = {sample(progs, session, 0, DBConfig(p_exploit=0.0), random.Random(s))[0].id for s in range(30)}
    assert seen == {"p0", "p1", "p2"}


def test_islands_round_robin_and_migration(lab):
    add(lab, "p1", "evaluated", 4.70, 0)
    add(lab, "p2", "evaluated", 4.60, 1)
    progs, session = ev.programs(lab["paths"]), ev.load_session(lab["paths"])
    cfg = DBConfig(migrate_every=2)
    assert next_island(progs, cfg) == 0
    assert maybe_migrate(progs, session, cfg)
    assert session["database"]["migrants"] == {"0": ["p2"], "1": ["p1"]}
    assert not maybe_migrate(progs, session, cfg)  # nothing new finished
    parent, _ = sample(progs, session, 0, DBConfig(p_exploit=1.0), random.Random(0))
    assert parent.id == "p2"  # the migrant is now island 0's best


# --- prompt --------------------------------------------------------------------------------


def test_prompt_contains_everything(lab):
    add(lab, "p1", "evaluated", 4.70, 0)
    add(lab, "p2", "rejected", None, 0).reason = "x"
    progs, session = ev.programs(lab["paths"]), ev.load_session(lab["paths"])
    from autolab.generate import llm_cfg

    text, meta = build_prompt(progs["p0"], [progs["p1"]], progs, session, ev.evolve_cfg(), llm_cfg(), lab["runs"],
                              random.Random(0))
    for needle in ("81,920,000 tokens", "quality champion", "Pareto frontier", "four dimensions", "# Prior programs", "## Program p1", "# Current program (p0)",
                   "- diagnosis **", "block `attention`", "masked_fill", "# What has been tried",
                   "# Recent rejections", "# Task", "`lr`: 1e-05 … 0.01"):
        assert needle in text, needle
    bar = 4.72 - 2 * 0.01
    assert f"{bar:.4f}" in text
    assert meta["parent"] == "p0" and meta["inspirations"] == ["p1"]


# --- LLM call failure modes ------------------------------------------------------------------


def fake_run(stdout="", returncode=0, stderr="", raise_timeout=False):
    def run(cmd, **kw):
        assert "--tools" in cmd and cmd[cmd.index("--tools") + 1] == ""
        assert kw["input"] == "PROMPT"
        if raise_timeout:
            raise subprocess.TimeoutExpired(cmd, 1)
        return subprocess.CompletedProcess(cmd, returncode, stdout, stderr)
    return run


GOOD = {"rationale": "use a shorter warmup", "expected_effect": "-0.01", "diffs": [], "hparams": {"warmup_steps": 32},
        "technique_ids": []}


@pytest.mark.parametrize("runner,needle", [
    (fake_run(raise_timeout=True), "timeout"),
    (fake_run(returncode=1, stderr="auth failed"), "exited 1: auth failed"),
    (fake_run(returncode=1, stdout=json.dumps({"result": "You've hit your weekly limit", "api_error_status": 429})),
     r"HTTP 429\): You've hit your weekly limit"),
    (fake_run(stdout="not json"), "not JSON"),
    (fake_run(stdout=json.dumps({"is_error": True, "result": "overloaded"})), "reported an error"),
    (fake_run(stdout=json.dumps({"structured_output": {"rationale": "x"}})), "fails the schema"),
])
def test_llm_failures(tmp_path, runner, needle):
    with pytest.raises(LLMError, match=needle):
        call("PROMPT", "SYS", REPLY_SCHEMA, "sonnet", {"claude_bin": "/bin/echo"}, tmp_path / "llm" / "s", "t",
             runner=runner)
    assert list((tmp_path / "llm" / "s").glob("*-t.json"))  # every call is logged
    assert (tmp_path / "llm_spend.jsonl").exists()


def test_llm_success(tmp_path):
    out = {"structured_output": GOOD, "total_cost_usd": 0.21, "modelUsage": {"claude-opus-5-5": {}}}
    r = call("PROMPT", "SYS", REPLY_SCHEMA, "opus", {"claude_bin": "/bin/echo"}, tmp_path / "llm" / "s", "t",
             runner=fake_run(stdout=json.dumps(out)))
    assert r["reply"] == GOOD and r["cost_usd"] == 0.21 and r["model"] == "claude-opus-5-5"
    spend = [json.loads(line) for line in (tmp_path / "llm_spend.jsonl").read_text().splitlines()]
    assert spend[-1]["usd"] == 0.21 and spend[-1]["ok"]


# --- generate_one ------------------------------------------------------------------------------


def caller_returning(reply=None, error=None):
    def caller(prompt, system, schema, model, cfg, log_dir, tag):
        assert "# Current program" in prompt and "evolutionary search" in system
        if error:
            raise LLMError(error)
        return {"reply": reply, "cost_usd": 0.2, "model": "claude-opus-5-5", "duration_s": 30.0, "log": "x.json"}
    return caller


def gen(lab, caller):
    return generate_one(paths=lab["paths"], rng=random.Random(1), log=lambda m: None, caller=caller,
                        model="opus", runs_dir=lab["runs"])


def test_generate_llm_child(lab):
    c = gen(lab, caller_returning(GOOD))
    assert (c.status, c.stage, c.created_by) == ("queued", "static", "claude-opus-5-5")
    assert c.hparams["warmup_steps"] == 32 and c.parent_id == "p0" and c.island == 0
    assert c.meta["cost_usd"] == 0.2 and c.meta["instruction"]


@pytest.mark.parametrize("caller,needle", [
    (caller_returning(error="timeout after 600s"), "timeout"),
    (caller_returning({**GOOD, "hparams": {}}), "changes nothing"),
    (caller_returning({**GOOD, "hparams": {"lr": 0.5}}), "out of range"),
])
def test_generate_falls_back_to_mutation(lab, caller, needle):
    c = gen(lab, caller)
    assert c.created_by == "mutation" and c.status == "queued" and needle in c.meta["fallback_reason"]
    assert "LLM fallback" in c.rationale


def test_generate_bad_diff_is_kept_and_falls_back(lab):
    bad = {**GOOD, "hparams": {}, "diffs": [{"search": "this text is not in the program", "replace": "x"}]}
    c = gen(lab, caller_returning(bad))
    progs = ev.programs(lab["paths"])
    rejected = [p for p in progs.values() if p.status == "rejected"]
    assert len(rejected) == 1 and "SEARCH text not found" in rejected[0].reason
    assert rejected[0].created_by == "claude-opus-5-5"
    assert c.created_by == "mutation" and "diffs don't apply" in c.meta["fallback_reason"]


def test_mutation_stays_in_range():
    spec = ev.evolve_cfg()["hparams"]
    parent = Program(id="p0", parent_id=None, base_commit="x", blocks={}, hparams=HP)
    from autolab.program import validate_hparams

    for s in range(200):
        patch, why = mutate(parent, spec, random.Random(s))
        assert patch and why.startswith("mutation:")
        assert not validate_hparams({**HP, **patch}, spec), (patch, validate_hparams({**HP, **patch}, spec))


# --- one real call (opt-in: AUTOLAB_REAL_LLM=1) --------------------------------------------------


@pytest.mark.skipif(os.environ.get("AUTOLAB_REAL_LLM") != "1", reason="real Claude call; set AUTOLAB_REAL_LLM=1")
def test_real_claude_call(lab):
    c = generate_one(paths=lab["paths"], rng=random.Random(0), log=print, model="sonnet", runs_dir=lab["runs"])
    assert c.created_by != "mutation", c.meta.get("fallback_reason")
    assert c.meta["cost_usd"] > 0 and c.rationale


def test_rate_limit_is_its_own_error(tmp_path):
    from autolab.llm import RateLimited

    run = fake_run(returncode=1, stdout=json.dumps({"result": "You've hit your weekly limit", "api_error_status": 429}))
    with pytest.raises(RateLimited):
        call("PROMPT", "SYS", REPLY_SCHEMA, "opus", {"claude_bin": "/bin/echo"}, tmp_path / "llm" / "s", "t", runner=run)


def test_rate_limit_stops_generation(lab):
    from autolab.llm import RateLimited

    def limited(*a, **k):
        raise RateLimited("claude exited 1 (HTTP 429): weekly limit")

    with pytest.raises(RateLimited):
        gen(lab, limited)
    assert set(ev.programs(lab["paths"])) == {"p0"}  # no fallback mutation queued


# --- research cards in proposals (M7) --------------------------------------------------------------------


CARD = {"name": "Decoupled weight decay on matrices only", "category": "optimizer", "component": "optimizer",
        "current": "AdamW on all params", "proposal": "wd 0.1 on 2-D weights", "mechanism": "limits norm growth",
        "evidence": [{"source": "Loshchilov & Hutter", "url": "https://arxiv.org/abs/1711.05101", "year": 2019,
                      "finding": "decoupled wd generalizes better"}],
        "expected_effect": "-0.02", "applicability": "multi-epoch overfitting", "risks": "slower early progress",
        "implementation": "param groups in build_optimizer"}


def test_cards_store_dedupe_and_bandit(tmp_path):
    from autolab import research

    added = research.add_cards([CARD, {**CARD, "name": "decoupled weight-decay on matrices ONLY"},
                                {**CARD, "name": "QK-norm", "category": "attention"}], {"session": "s"})
    assert [c["id"] for c in added] == ["c1", "c2"] and added[1]["name"] == "QK-norm"
    cards = research.load_cards()
    progs = [("s", Program(id="p1", parent_id="p0", base_commit="x", blocks={}, hparams=HP, status="accepted",
                           meta={"technique_ids": ["c1"]})),
             ("s", Program(id="p2", parent_id="p0", base_commit="x", blocks={}, hparams=HP, status="rejected",
                           meta={"technique_ids": ["c2"]}))]
    stats = research.card_stats(cards, progs)
    assert stats["c1"]["outcomes"] == {"accepted": 1} and stats["c2"]["mean_reward"] == 0.0
    research.add_cards([{**CARD, "name": "Muon optimizer"}], {"session": "s"})
    picked = research.pick_cards(2, random.Random(0), research.load_cards(), research.card_stats(research.load_cards(), progs))
    assert [c["id"] for c in picked] == ["c3", "c1"]  # untried first, then the winner over the loser


def test_directed_child_applies_and_credits_the_card(lab):
    from autolab import research

    card = research.add_cards([CARD], {"session": "s"})[0]
    seen = {}

    def caller(prompt, system, schema, model, cfg, log_dir, tag, **kw):
        seen["prompt"] = prompt
        return {"reply": {**GOOD, "hparams": {"weight_decay": 0.1}, "technique_ids": []}, "cost_usd": 0.1,
                "model": "claude-opus-5-5", "duration_s": 1.0, "log": "x"}

    c = generate_one(paths=lab["paths"], rng=random.Random(0), log=lambda m: None, caller=caller, model="opus",
                     runs_dir=lab["runs"], card=card)
    assert "Apply technique card **c1**" in seen["prompt"] and "# Relevant techniques" in seen["prompt"]
    assert c.meta["technique_ids"] == ["c1"] and c.meta["instruction"] == "research" and c.parent_id == "p0"


def test_explore_handles_mixed_cell_kinds(lab):
    """Older programs sit in (params, tps) cells and M8 ones in ("ctx", context, latency) cells; picking a random
    cell once crashed every such cycle (TypeError comparing str and int)."""
    add(lab, "p1", "evaluated", 4.70, 0)
    p = add(lab, "p2", "evaluated", 4.69, 0)
    p.scores["metrics"] = {"context": 256, "decode_ms_per_token": 7.0}
    save(p, lab["paths"].programs)
    progs, session = ev.programs(lab["paths"]), ev.load_session(lab["paths"])
    seen = {sample(progs, session, 0, DBConfig(p_exploit=0.0, p_frontier=0.0), random.Random(s))[0].id for s in range(20)}
    assert "p2" in seen and "p1" in seen
