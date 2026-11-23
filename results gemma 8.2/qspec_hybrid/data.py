from __future__ import annotations

import base64
import io
import json
import math
import random
from collections import defaultdict
from pathlib import Path

from .core import (
    MODEL,
    PROTOCOL,
    SEED,
    TASKS,
    atomic_json,
    digest,
    file_hash,
    read_json,
)
from .upstream import Scorers, prepare as prepare_scorers, selected

DATASETS = {
    "textvqa": ("lmms-lab-encoder/textvqa", "validation"),
    "chartqa": ("lmms-lab-encoder/ChartQA", "test"),
    "mmstar": ("Lin-Chen/MMStar", "val"),
    "omnidoc_ocr": ("opendatalab/OmniDocBench", "annotations"),
    "calibration": ("derek-thomas/ScienceQA", "train"),
}


def image_of(value):
    from PIL import Image

    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, bytes):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        if len(value) < 300 and Path(value).is_file():
            return Image.open(value).convert("RGB")
        return Image.open(io.BytesIO(base64.b64decode(value))).convert("RGB")
    raise ValueError(f"Unsupported image representation: {type(value).__name__}")


def balanced(rows, count, seed=SEED):
    """Round-robin strata after seeded shuffle, without response-dependent filtering."""
    buckets = defaultdict(list)
    for row in rows:
        buckets[row["stratum"]].append(row)
    rng = random.Random(seed)
    for items in buckets.values():
        rng.shuffle(items)
    keys = sorted(buckets)
    output = []
    while len(output) < count and any(buckets.values()):
        for key in keys:
            if buckets[key] and len(output) < count:
                output.append(buckets[key].pop())
    if len(output) < count:
        raise ValueError(
            f"Only {len(output)} eligible independent images; need {count}"
        )
    return output


def revisions(root):
    from huggingface_hub import HfApi

    path = root / "revisions.json"
    if path.exists():
        return read_json(path)
    api = HfApi()
    value = {
        "model": {"id": MODEL, "revision": api.model_info(MODEL).sha},
        "datasets": {},
    }
    for key, (repo, split) in DATASETS.items():
        value["datasets"][key] = {
            "id": repo,
            "revision": api.dataset_info(repo).sha,
            "split": split,
        }
    atomic_json(path, value)
    return value


def prompt(root, task, doc):
    if task == "textvqa":
        return selected(root, task, ["textvqa_doc_to_text"])["textvqa_doc_to_text"](
            doc,
            {
                "pre_prompt": "",
                "post_prompt": "\nAnswer the question using a single word or phrase.",
                "ocr": False,
            },
        )
    if task == "chartqa":
        return selected(root, task, ["chartqa_doc_to_text"])["chartqa_doc_to_text"](
            doc,
            {
                "pre_prompt": "",
                "post_prompt": "\nAnswer the question with a single word.",
            },
        )
    if task == "mmstar":
        return selected(root, task, ["mmstar_doc_to_text"])["mmstar_doc_to_text"](
            doc,
            {
                "pre_prompt": "",
                "post_prompt": "\nAnswer with the option's letter from the given choices directly",
            },
        )
    return "Transcribe all text in this cropped document paragraph exactly. Output only the transcription, without commentary or Markdown fences."


def save_case(root, task, doc, image, target, identity, stratum, group):
    image = image.convert("RGB")
    # Hash decoded pixels, not container metadata; identical images cannot cross splits.
    import hashlib

    image_digest = hashlib.sha256(
        str(image.size).encode() + image.tobytes()
    ).hexdigest()
    dest = root / "data" / "images" / f"{image_digest}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        image.save(dest)
    return {
        "id": f"{task}/{identity}",
        "task": task,
        "group": str(group),
        "stratum": str(stratum),
        "image": str(dest.relative_to(root)),
        "image_sha256": image_digest,
        "image_file_sha256": file_hash(dest),
        "target": target,
        "prompt": prompt(root, task, doc),
        "max_tokens": PROTOCOL["max_tokens"].get(task, 512),
    }


def retain_unique(cases, case, seen, exclusions):
    if case["image_sha256"] in seen:
        exclusions.append(
            {
                "id": case["id"],
                "reason": "duplicate decoded image, removed before any generation",
                "image_sha256": case["image_sha256"],
            }
        )
        return
    seen.add(case["image_sha256"])
    cases.append(case)


def prepare(root):
    from datasets import Image as HFImage, load_dataset
    from huggingface_hub import hf_hub_download

    path = root / "data" / "prepared.json"
    prepare_scorers(root)
    Scorers(root).parity_check()
    if path.exists():
        value = read_json(path)
        if value["protocol"] != digest(PROTOCOL):
            raise RuntimeError("Prepared inputs belong to a different protocol")
        if value["input_hash"] != digest(
            {k: v for k, v in value.items() if k != "input_hash"}
        ):
            raise RuntimeError(
                "Frozen input manifest checksum failed; restore an intact backup"
            )
        for split in ("development", "evaluation", "heldout", "calibration"):
            for case in value[split]:
                image = root / case["image"]
                if not image.exists() or file_hash(image) != case["image_file_sha256"]:
                    raise RuntimeError(f"Frozen image missing or changed: {case['id']}")
        return value
    rev = revisions(root)
    result = {
        "protocol": digest(PROTOCOL),
        "revisions": rev,
        "development": [],
        "evaluation": [],
        "heldout": [],
        "calibration": [],
    }
    seen_images, exclusions = set(), []
    required = PROTOCOL["pilot_samples_per_task"] + 200 + 40
    for task in TASKS:
        print(f"Preparing frozen {task} samples", flush=True)
        info = rev["datasets"][task]
        candidates = []
        if task != "omnidoc_ocr":
            dataset = load_dataset(
                info["id"], split=info["split"], revision=info["revision"]
            )
            if "image" not in dataset.column_names:
                raise ValueError(
                    f"Image column missing in {task}: {dataset.column_names}"
                )
            if isinstance(dataset.features["image"], HFImage):
                dataset = dataset.cast_column("image", HFImage(decode=False))
            groups = set()
            for index, doc in enumerate(dataset):
                raw = doc["image"]
                import hashlib

                if doc.get("image_id") is not None:
                    group = str(doc["image_id"])
                elif isinstance(raw, dict) and raw.get("bytes"):
                    group = hashlib.sha256(raw["bytes"]).hexdigest()
                else:
                    group = str(raw.get("path") if isinstance(raw, dict) else index)
                if group in groups:
                    continue
                groups.add(group)
                stratum = str(doc.get("category", doc.get("type", "all")))
                candidates.append({"index": index, "stratum": stratum, "group": group})
            chosen = balanced(candidates, min(len(candidates), required + 100))
            cases = []
            for item in chosen:
                doc = dict(dataset[item["index"]])
                target = doc["answers"] if task == "textvqa" else doc["answer"]
                if task == "textvqa" and len(target) != 10:
                    raise ValueError("TextVQA row has missing human answers")
                if isinstance(target, list) and task != "textvqa":
                    if len(target) != 1:
                        raise ValueError(f"Unexpected target schema for {task}")
                    target = target[0]
                if target is None or (isinstance(target, str) and not target.strip()):
                    raise ValueError(
                        f"Unlabelled selected document: {task}/{item['index']}"
                    )
                if task == "mmstar" and str(target).strip().upper() not in "ABCD":
                    raise ValueError("Unexpected MMStar answer vocabulary")
                identity = str(
                    doc.get(
                        "question_id", doc.get("index", doc.get("id", item["index"]))
                    )
                )
                case = save_case(
                    root,
                    task,
                    doc,
                    image_of(doc["image"]),
                    target,
                    identity + "@row" + str(item["index"]),
                    item["stratum"],
                    item["group"],
                )
                case.update(source_id=identity, source_row_index=item["index"])
                retain_unique(cases, case, seen_images, exclusions)
                if len(cases) == required:
                    break
        else:
            annotations = Path(
                hf_hub_download(
                    info["id"],
                    "OmniDocBench.json",
                    repo_type="dataset",
                    revision=info["revision"],
                )
            )
            pages = json.loads(annotations.read_text())
            for index, page in enumerate(pages):
                blocks = [
                    b
                    for b in page.get("layout_dets", [])
                    if b.get("category_type") == "text_block"
                    and not b.get("ignore")
                    and len(str(b.get("text", ""))) >= 100
                    and b.get("attribute", {}).get("text_rotate", "normal")
                    in ("normal", "horizontal")
                    and len(b.get("poly", [])) == 8
                ]
                if not blocks:
                    continue
                block = max(blocks, key=lambda x: len(x["text"]))
                attr = page["page_info"].get("page_attribute", {})
                candidates.append(
                    {
                        "index": index,
                        "block": block,
                        "group": page["page_info"]["image_path"],
                        "stratum": str(attr.get("language", "unknown"))
                        + "/"
                        + str(attr.get("data_source", "unknown")),
                    }
                )
            cases = []
            files = None
            for item in balanced(candidates, min(len(candidates), required + 100)):
                page = pages[item["index"]]
                name = Path(page["page_info"]["image_path"]).name
                filename = f"images/{name}"
                try:
                    local = hf_hub_download(
                        info["id"],
                        filename,
                        repo_type="dataset",
                        revision=info["revision"],
                    )
                except Exception as error:
                    # Resolve naming only; never replace a selected example with another.
                    from huggingface_hub import HfApi

                    files = files or HfApi().list_repo_files(
                        info["id"], repo_type="dataset", revision=info["revision"]
                    )
                    matches = [f for f in files if f.endswith("/" + name)]
                    if len(matches) != 1:
                        raise RuntimeError(
                            f"Cannot resolve selected OmniDocBench image {name}"
                        ) from error
                    local = hf_hub_download(
                        info["id"],
                        matches[0],
                        repo_type="dataset",
                        revision=info["revision"],
                    )
                image = image_of(str(local))
                points = item["block"]["poly"]
                box = (
                    max(0, math.floor(min(points[0::2]))),
                    max(0, math.floor(min(points[1::2]))),
                    min(image.width, math.ceil(max(points[0::2]))),
                    min(image.height, math.ceil(max(points[1::2]))),
                )
                if box[2] <= box[0] or box[3] <= box[1]:
                    raise ValueError(f"Invalid selected OCR region: {name}")
                case = save_case(
                    root,
                    task,
                    {},
                    image.crop(box),
                    item["block"]["text"],
                    name + ":" + str(item["block"].get("anno_id")),
                    item["stratum"],
                    name,
                )
                retain_unique(cases, case, seen_images, exclusions)
                if len(cases) == required:
                    break
        if len(cases) < required:
            raise RuntimeError(
                f"Insufficient unique eligible inputs for {task}; no response-based sample replacement was performed"
            )
        for n, case in enumerate(cases):
            destination = (
                "development" if n < 6 else "evaluation" if n < 206 else "heldout"
            )
            result[destination].append(case)
    # Independent image/text calibration corpus, frozen once for every export.
    info = rev["datasets"]["calibration"]
    science = load_dataset(info["id"], split=info["split"], revision=info["revision"])
    science = science.cast_column("image", HFImage(decode=False))
    for index in random.Random(SEED).sample(range(len(science)), len(science)):
        doc = science[index]
        if not doc.get("image") or not (
            doc["image"].get("bytes") or doc["image"].get("path")
        ):
            continue
        image = image_of(doc["image"])
        target = str(doc["choices"][int(doc["answer"])])
        case = save_case(
            root,
            "calibration",
            {},
            image,
            target,
            str(index),
            "ScienceQA/train",
            str(index),
        )
        case["prompt"] = (
            doc["question"] + "\n" + "\n".join(str(x) for x in doc["choices"])
        )
        if case["image_sha256"] in seen_images:
            continue
        seen_images.add(case["image_sha256"])
        result["calibration"].append(case)
        if len(result["calibration"]) == PROTOCOL["calibration_samples"]:
            break
    if len(result["calibration"]) != PROTOCOL["calibration_samples"]:
        raise RuntimeError("Insufficient disjoint calibration images")
    for task in TASKS:
        group = [x for x in result["evaluation"] if x["task"] == task]
        selected_ids = {x["id"] for x in balanced(group, 40, SEED + 1)}
        for case in group:
            case["timing_repeat"] = case["id"] in selected_ids
    ids = [
        c["id"]
        for part in ("development", "evaluation", "heldout", "calibration")
        for c in result[part]
    ]
    if len(ids) != len(set(ids)):
        raise RuntimeError(
            "Duplicate frozen request IDs; no model requests were started"
        )
    result["input_hash"] = digest(result)
    atomic_json(root / "data" / "input_selection_exclusions.json", exclusions)
    atomic_json(path, result)
    return result
