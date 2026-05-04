"""Analyze why t_raw_true images were excluded by the prefilter."""
import csv
from pathlib import Path
from collections import defaultdict, Counter

CACHE = Path("data/.open_images_cache")
FO_ROOT = Path.home() / "fiftyone" / "open-images-v7"

with open(CACHE / "t_raw_true_ids.txt") as f:
    t_ids = set(line.strip() for line in f if line.strip())

classes_csv = FO_ROOT / "test" / "metadata" / "classes.csv"
name_to_id = {}
id_to_name = {}
with open(classes_csv, newline="", encoding="utf-8") as f:
    for row in csv.reader(f):
        if len(row) >= 2:
            id_to_name[row[0]] = row[1]
            name_to_id[row[1]] = row[0]

face_label_id = name_to_id["Human face"]
person_label_id = name_to_id["Person"]

# split ごとに独立して集計（最後にマージ）
split_reasons: dict[str, dict[str, str]] = {}
split_stats: dict[str, dict] = {}

for split in ["test", "validation"]:
    ann_csv = FO_ROOT / split / "labels" / "detections.csv"
    if not ann_csv.exists():
        continue

    image_labels = defaultdict(set)
    image_faces = defaultdict(list)
    with open(ann_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_id = row["ImageID"]
            image_labels[img_id].add(row["LabelName"])
            if row["LabelName"] == face_label_id:
                image_faces[img_id].append(row)

    reasons: dict[str, str] = {}
    pass_count = 0
    fail_no_person = 0
    fail_quality = 0
    in_split = 0

    for img_id in t_ids:
        if img_id not in image_labels:
            continue  # このsplitにいない画像はスキップ
        in_split += 1
        labels = image_labels[img_id]

        if person_label_id not in labels:
            reasons[img_id] = "NO_PERSON"
            fail_no_person += 1
            continue

        valid_faces = []
        fail_flags: list[str] = []
        for r in image_faces.get(img_id, []):
            failed = False
            for flag in ("IsDepiction", "IsGroupOf", "IsInside", "IsOccluded", "IsTruncated"):
                if int(r.get(flag, 0)):
                    fail_flags.append(flag)
                    failed = True
                    break
            if failed:
                continue
            w = float(r["XMax"]) - float(r["XMin"])
            h = float(r["YMax"]) - float(r["YMin"])
            if w * h < 0.005:
                fail_flags.append("small_face")
                continue
            aspect = w / h if h > 0 else 999.0
            if not (0.1 <= aspect <= 10.0):
                fail_flags.append("bad_aspect")
                continue
            valid_faces.append(r)

        if not valid_faces:
            unique_flags = sorted(set(fail_flags))
            reasons[img_id] = "QUALITY(" + ",".join(unique_flags) + ")"
            fail_quality += 1
        elif len(valid_faces) > 4:
            reasons[img_id] = "TOO_MANY_FACES"
        else:
            reasons[img_id] = "PASS"
            pass_count += 1

    split_reasons[split] = reasons
    split_stats[split] = {
        "in_split": in_split,
        "PASS": pass_count,
        "NO_PERSON": fail_no_person,
        "QUALITY": fail_quality,
    }

for split in ["test", "validation"]:
    s = split_stats[split]
    print(f"=== {split} ({s['in_split']} images in split) ===")
    print(f"  PASS:            {s['PASS']}")
    print(f"  NO_PERSON:       {s['NO_PERSON']}")
    print(f"  QUALITY fail:    {s['QUALITY']}")
    print()

# 全体マージ（各画像はどちらか一方のsplitにのみ存在）
merged: dict[str, str] = {}
for split in ["test", "validation"]:
    merged.update(split_reasons[split])

print("=== 全体サマリ (480 images) ===")
all_reasons = Counter(merged.values())
for reason, count in all_reasons.most_common():
    print(f"  {reason}: {count}")

# NO_PERSON画像に他にどんなラベルがあるか
print()
print("=== NO_PERSON 画像のラベル分布（上位20） ===")
no_person_label_counter = Counter()
for split in ["test", "validation"]:
    ann_csv = FO_ROOT / split / "labels" / "detections.csv"
    if not ann_csv.exists():
        continue
    img_labels_map = defaultdict(set)
    with open(ann_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            img_labels_map[row["ImageID"]].add(row["LabelName"])
    for img_id, reason in merged.items():
        if reason == "NO_PERSON" and img_id in img_labels_map:
            for lbl in img_labels_map[img_id]:
                no_person_label_counter[id_to_name.get(lbl, lbl)] += 1

for name, count in no_person_label_counter.most_common(20):
    print(f"  {name}: {count}")
