"""Narrow, immutable upstream scorer loading; no editable package metadata assumptions."""

from __future__ import annotations

import ast
import importlib.util
import json
import re
import urllib.request
from pathlib import Path

from .core import LMMS, OMNI, atomic_json, file_hash

FILES = {
    "vqa": (
        "EvolvingLMMs-Lab/lmms-eval",
        LMMS,
        "lmms_eval/tasks/_task_utils/vqa_eval_metric.py",
    ),
    "textvqa": ("EvolvingLMMs-Lab/lmms-eval", LMMS, "lmms_eval/tasks/textvqa/utils.py"),
    "chartqa": ("EvolvingLMMs-Lab/lmms-eval", LMMS, "lmms_eval/tasks/chartqa/utils.py"),
    "mcq": (
        "EvolvingLMMs-Lab/lmms-eval",
        LMMS,
        "lmms_eval/tasks/_task_utils/mcq_extract.py",
    ),
    "mmstar": ("EvolvingLMMs-Lab/lmms-eval", LMMS, "lmms_eval/tasks/mmstar/utils.py"),
    "omni_metric": ("opendatalab/OmniDocBench", OMNI, "src/metrics/cal_metric.py"),
}


def prepare(root):
    folder = root / "sources"
    folder.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for key, (repo, commit, path) in FILES.items():
        dest = folder / f"{key}.py"
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        if not dest.exists():
            with urllib.request.urlopen(url, timeout=120) as response:
                data = response.read()
            compile(data, url, "exec")
            dest.write_bytes(data)
        receipts[key] = {"url": url, "sha256": file_hash(dest)}
    old = folder / "receipts.json"
    if old.exists() and json.loads(old.read_text()) != receipts:
        raise RuntimeError(
            "Pinned evaluator files changed; refusing to mix scoring revisions"
        )
    atomic_json(old, receipts)


def selected(root: Path, key: str, names: list[str], namespace=None):
    """Keep named definitions verbatim, excluding optional imports/decorators.

    This avoids importing unrelated table/formula/LLM evaluator dependencies.
    Source text and file hashes are preserved in sources/receipts.json.
    """
    path = root / "sources" / f"{key}.py"
    tree = ast.parse(path.read_text())
    nodes = [
        n
        for n in tree.body
        if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names
    ]
    if {n.name for n in nodes} != set(names):
        raise RuntimeError(f"Required pinned definitions missing: {key}/{names}")
    for node in nodes:
        node.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    literals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    literals[target.id] = value
    space = {"re": re, **literals, **(namespace or {})}
    exec(compile(module, str(path), "exec"), space)
    return space


class Scorers:
    def __init__(self, root: Path):
        self.root = root
        self.vqa = selected(root, "vqa", ["EvalAIAnswerProcessor"])[
            "EvalAIAnswerProcessor"
        ]()
        self.relaxed = selected(root, "chartqa", ["relaxed_correctness"])[
            "relaxed_correctness"
        ]
        path = root / "sources" / "mcq.py"
        spec = importlib.util.spec_from_file_location("qspec_pinned_mcq", path)
        module = importlib.util.module_from_spec(spec)
        import sys

        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        self.mcq = module.extract_mcq_answer

    def score(self, case, prediction):
        task, target = case["task"], case["target"]
        if task == "textvqa":
            answers = [self.vqa(str(x)) for x in target]
            if len(answers) != 10:
                raise ValueError("TextVQA requires all ten human answers")
            parsed = self.vqa(prediction)
            scores = [
                min(1.0, sum(a == parsed for j, a in enumerate(answers) if j != i) / 3)
                for i in range(len(answers))
            ]
            return {
                "metric": "vqa_consensus",
                "value": sum(scores) / len(scores),
                "parsed": parsed,
                "higher_is_better": True,
            }
        if task == "chartqa":
            return {
                "metric": "relaxed_accuracy",
                "value": float(self.relaxed(prediction, str(target))),
                "parsed": prediction,
                "higher_is_better": True,
            }
        if task == "mmstar":
            parsed = self.mcq(prediction, choices=list("ABCD"))
            return {
                "metric": "mc_accuracy",
                "value": float(parsed == str(target).strip().upper()),
                "parsed": parsed,
                "higher_is_better": True,
            }
        if task == "omnidoc_ocr":
            import Levenshtein

            # Raw-text edit/max-length is exactly the metric's per-block formula.
            # No page matcher, table score or full-benchmark overall is claimed.
            gt, pred = str(target), prediction
            value = Levenshtein.distance(gt, pred) / max(len(gt), len(pred), 1)
            return {
                "metric": "raw_text_block_normalized_edit_distance",
                "value": value,
                "parsed": pred,
                "higher_is_better": False,
            }
        raise ValueError(task)

    def parity_check(self):
        """Exercise the official OmniDocBench class against our per-block formula."""
        import os
        import tempfile
        import Levenshtein
        import pandas as pd

        ns = selected(
            self.root,
            "omni_metric",
            ["_as_sample_list", "call_Edit_dist"],
            {"Levenshtein": Levenshtein, "pd": pd, "json": json},
        )
        original = Path.cwd()
        with tempfile.TemporaryDirectory(prefix="qspec-scorer-") as temporary:
            try:
                os.chdir(temporary)
                Path("result").mkdir()
                for gt, pred in [("abc", "axc"), ("a", ""), ("中文段落", "中文落")]:
                    sample = {"img_id": "check.png", "gt": gt, "pred": pred}
                    rows, _ = ns["call_Edit_dist"]([sample]).evaluate()
                    actual = self.score({"task": "omnidoc_ocr", "target": gt}, pred)[
                        "value"
                    ]
                    assert abs(rows[0]["metric"]["Edit_dist"] - actual) < 1e-12
            finally:
                os.chdir(original)
        assert (
            self.score({"task": "textvqa", "target": ["yes"] * 3 + ["no"] * 7}, "yes")[
                "value"
            ]
            == 0.9
        )
        assert self.relaxed("105", "100") and not self.relaxed("106", "100")
        assert self.mcq("B", choices=list("ABCD")) == "B"
        return {
            "passed": True,
            "omni_scope": "raw cropped-text OCR NED, not page parsing",
        }
