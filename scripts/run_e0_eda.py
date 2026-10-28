#!/usr/bin/env python3
"""MIND-small E0: temporal EDA, A.2/A.3 tables, R1-R5 and 3 basic graph snapshots.

Default graph snapshots (per proposal v2):
  Ghist             = History-derived edges only + content metadata relations
  Ghist_click_0913  = Ghist + positive impression clicks on 2019-11-09..2019-11-13
  Ghist_click_0914  = Ghist + positive impression clicks on 2019-11-09..2019-11-14

The script intentionally does NOT use dev/test click labels to construct graph snapshots.
It uses train histories as the pre-2019-11-09 seed, then adds labeled clicks from train
impressions strictly before each snapshot's cutoff date.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import pandas as pd

BEHAVIOR_COLS = ["impression_id", "user_id", "time", "history", "impressions"]
NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entities", "abstract_entities"]
BASE_URL = "https://mind201910small.blob.core.windows.net/release"
TRAIN_URL = f"{BASE_URL}/MINDsmall_train.zip"
DEV_URL = f"{BASE_URL}/MINDsmall_dev.zip"
DATE_START = pd.Timestamp("2019-11-09")
DATE_END = pd.Timestamp("2019-11-15")


def parse_space_ids(value: object) -> List[str]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    s = str(value).strip()
    return s.split() if s else []


def parse_impressions(value: object) -> List[Tuple[str, int]]:
    """Parse `N123-1 N456-0`; right-split so IDs with hyphens remain safe."""
    out: List[Tuple[str, int]] = []
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return out
    for token in str(value).strip().split():
        try:
            news_id, label = token.rsplit("-", 1)
            if label in {"0", "1"} and news_id:
                out.append((news_id, int(label)))
        except ValueError:
            continue
    return out


def parse_entities(value: object) -> List[Tuple[str, float]]:
    """Return unique (WikidataId, confidence) pairs from one MIND entity JSON field."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return []
    s = str(value).strip()
    if not s:
        return []
    try:
        items = json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(items, list):
        return []
    found: Dict[str, float] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        entity_id = item.get("WikidataId") or item.get("WikidataID") or item.get("wikidata_id")
        if entity_id is None or not str(entity_id).strip():
            continue
        entity_id = str(entity_id).strip()
        raw_conf = item.get("Confidence", item.get("confidence", 1.0))
        try:
            confidence = float(raw_conf)
        except (TypeError, ValueError):
            confidence = 1.0
        if not math.isfinite(confidence):
            confidence = 1.0
        found[entity_id] = max(found.get(entity_id, 0.0), confidence)
    return list(found.items())


def find_named_file(root: Path, filename: str) -> Path:
    if not root.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục: {root}")
    matches = list(root.rglob(filename)) if root.is_dir() else ([root] if root.name == filename else [])
    if not matches:
        raise FileNotFoundError(f"Không tìm thấy {filename} bên trong {root}")
    # Prefer shallow paths if archive nested the files one folder too deep.
    return sorted(matches, key=lambda p: (len(p.parts), str(p)))[0]


def read_behaviors(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=BEHAVIOR_COLS,
                     dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                     on_bad_lines="warn")
    if len(df.columns) != 5:
        raise ValueError(f"{path} không có đúng 5 cột behaviors.tsv")
    df["source_split"] = source
    # MIND timestamps: MM/DD/YYYY HH:MM:SS AM/PM
    df["datetime"] = pd.to_datetime(df["time"], format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    bad = int(df["datetime"].isna().sum())
    if bad:
        print(f"[WARN] {path}: không parse được {bad} timestamp; các hàng đó được bỏ khỏi bucket thời gian.")
    df["date"] = df["datetime"].dt.strftime("%Y-%m-%d")
    return df


def read_news(path: Path, source: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", header=None, names=NEWS_COLS,
                     dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE,
                     on_bad_lines="warn")
    if len(df.columns) != 8:
        raise ValueError(f"{path} không có đúng 8 cột news.tsv (MIND hiện có 8 trường)")
    df["source_split"] = source
    return df


def count_clicks_and_candidates(behaviors: pd.DataFrame) -> Tuple[int, int]:
    clicks = candidates = 0
    for value in behaviors["impressions"].tolist():
        parsed = parse_impressions(value)
        candidates += len(parsed)
        clicks += sum(label for _, label in parsed)
    return clicks, candidates


def make_a2(beh_train: pd.DataFrame, beh_dev: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    train = beh_train.copy()
    dev = beh_dev.copy()
    train_days = train["date"].between("2019-11-09", "2019-11-14")
    dev_days = dev["date"].eq("2019-11-15")
    combined = pd.concat([train.loc[train_days], dev.loc[dev_days]], ignore_index=True)
    specs = [
        ("Train (chọn tham số)", "09-13/11/2019", combined[combined["date"].between("2019-11-09", "2019-11-13")], "MINDsmall_train/behaviors.tsv"),
        ("Validation", "14/11/2019", combined[combined["date"].eq("2019-11-14")], "MINDsmall_train/behaviors.tsv"),
        ("Train cuối", "09-14/11/2019", combined[combined["date"].between("2019-11-09", "2019-11-14")], "MINDsmall_train/behaviors.tsv"),
        ("Test", "15/11/2019", combined[combined["date"].eq("2019-11-15")], "MINDsmall_dev/behaviors.tsv"),
    ]
    rows = []
    for split_name, days, subset, source in specs:
        clicks, candidates = count_clicks_and_candidates(subset)
        rows.append({
            "Split": split_name,
            "Ngày": days,
            "# Impression": int(len(subset)),
            "# Click": int(clicks),
            "# User": int(subset["user_id"].nunique()),
            "# Candidate occurrences": int(candidates),
            "# Tin ứng viên duy nhất": int(len({nid for v in subset["impressions"] for nid, _ in parse_impressions(v)})),
            "Nguồn số liệu": source,
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / "A2_split_statistics.csv", index=False, encoding="utf-8-sig")
    return out


def get_history_seed(beh_train: pd.DataFrame) -> Tuple[Dict[str, List[str]], Set[Tuple[str, str]]]:
    """Use earliest train behavior per user to avoid later History values leaking forward."""
    valid = beh_train[beh_train["datetime"].notna()].sort_values(["datetime", "impression_id"])
    first_rows = valid.drop_duplicates("user_id", keep="first")
    sequences: Dict[str, List[str]] = {}
    pairs: Set[Tuple[str, str]] = set()
    for row in first_rows.itertuples(index=False):
        seq = parse_space_ids(getattr(row, "history"))
        uid = str(getattr(row, "user_id"))
        sequences[uid] = seq
        for nid in seq:
            pairs.add((uid, nid))
    return sequences, pairs


def make_history_consistency(beh_all: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for uid, group in beh_all.groupby("user_id", sort=False):
        seqs = set()
        for h in group["history"].tolist():
            seqs.add(" ".join(parse_space_ids(h)))
        rows.append({
            "user_id": uid,
            "# Impression": int(len(group)),
            "# History variants": int(len(seqs)),
            "History changes": bool(len(seqs) > 1),
        })
    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "history_consistency_by_user.csv", index=False, encoding="utf-8-sig")
    summary = pd.DataFrame([{
        "# User": int(len(result)),
        "# User có History thay đổi": int(result["History changes"].sum()) if len(result) else 0,
        "% User có History thay đổi": round(float(result["History changes"].mean() * 100), 4) if len(result) else 0.0,
        "Ghi chú": "So sánh chuỗi History giữa các impression của cùng user; graph Ghist seed chỉ dùng History ở impression train sớm nhất của mỗi user.",
    }])
    summary.to_csv(out_dir / "history_consistency_summary.csv", index=False, encoding="utf-8-sig")
    return summary


def make_bucket_eda(beh_all: pd.DataFrame, beh_train: pd.DataFrame, out_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create daily and 6-hour EDA, with candidate cold-start measured before each bucket."""
    valid = beh_all[beh_all["datetime"].notna() & beh_all["date"].between("2019-11-09", "2019-11-15")].copy()
    valid = valid.sort_values(["datetime", "impression_id"])
    # Seed global previously-clicked set from earliest train histories only (per proposal's pre-window-history assumption).
    hist_seq, _ = get_history_seed(beh_train)
    prior_clicked: Set[str] = {nid for seq in hist_seq.values() for nid in seq}
    first_seen = valid.groupby("user_id")["datetime"].min().to_dict()

    daily_rows = []
    for day in pd.date_range("2019-11-09", "2019-11-15", freq="D"):
        day_s = day.strftime("%Y-%m-%d")
        bucket = valid[valid["date"].eq(day_s)]
        if bucket.empty:
            daily_rows.append({"Ngày": day_s, "# Imp.": 0, "# Click": 0, "CTR (%)": 0.0, "# User": 0,
                               "% user mới": 0.0, "# Tin ứng viên": 0, "% tin ứng viên chưa từng được click": 0.0,
                               "# Candidate occurrences": 0})
            continue
        day_candidates: Set[str] = set()
        day_clicks: Set[str] = set()
        click_count = candidate_count = 0
        for row in bucket.itertuples(index=False):
            for nid, label in parse_impressions(getattr(row, "impressions")):
                day_candidates.add(nid)
                candidate_count += 1
                if label == 1:
                    click_count += 1
                    day_clicks.add(nid)
        unseen = sum(1 for nid in day_candidates if nid not in prior_clicked)
        users = set(bucket["user_id"].astype(str))
        new_users = sum(1 for uid in users if first_seen.get(uid) is not None and first_seen[uid].strftime("%Y-%m-%d") == day_s)
        daily_rows.append({
            "Ngày": day_s,
            "# Imp.": int(len(bucket)),
            "# Click": int(click_count),
            "CTR (%)": round(100.0 * click_count / candidate_count, 6) if candidate_count else 0.0,
            "# User": int(len(users)),
            "% user mới": round(100.0 * new_users / len(users), 6) if users else 0.0,
            "# Tin ứng viên": int(len(day_candidates)),
            "% tin ứng viên chưa từng được click": round(100.0 * unseen / len(day_candidates), 6) if day_candidates else 0.0,
            "# Candidate occurrences": int(candidate_count),
        })
        # Update only after calculating current-day new/cold-start metrics: clicks in this day are prior for tomorrow, not today.
        prior_clicked.update(day_clicks)
    daily = pd.DataFrame(daily_rows)
    daily.to_csv(out_dir / "A3_daily_eda.csv", index=False, encoding="utf-8-sig")

    # Repeat for 6-hour buckets with a fresh initial seed.
    hist_seq, _ = get_history_seed(beh_train)
    prior_clicked_6h: Set[str] = {nid for seq in hist_seq.values() for nid in seq}
    valid["bucket_6h"] = valid["datetime"].dt.floor("6h")
    first_seen_6h = valid.groupby("user_id")["datetime"].min().to_dict()
    six_rows = []
    for bucket_start, bucket in valid.groupby("bucket_6h", sort=True):
        candidates_set: Set[str] = set()
        click_set: Set[str] = set()
        clicks = n_candidates = 0
        for row in bucket.itertuples(index=False):
            for nid, label in parse_impressions(getattr(row, "impressions")):
                candidates_set.add(nid)
                n_candidates += 1
                if label == 1:
                    clicks += 1
                    click_set.add(nid)
        users = set(bucket["user_id"].astype(str))
        new_users = sum(1 for uid in users if first_seen_6h.get(uid) is not None and first_seen_6h[uid].floor("6h") == bucket_start)
        unseen = sum(1 for nid in candidates_set if nid not in prior_clicked_6h)
        six_rows.append({
            "Bucket 6h bắt đầu": bucket_start.strftime("%Y-%m-%d %H:%M:%S"),
            "# Imp.": int(len(bucket)),
            "# Click": int(clicks),
            "CTR (%)": round(100.0 * clicks / n_candidates, 6) if n_candidates else 0.0,
            "# User": int(len(users)),
            "% user mới": round(100.0 * new_users / len(users), 6) if users else 0.0,
            "# Tin ứng viên": int(len(candidates_set)),
            "% tin ứng viên chưa từng được click": round(100.0 * unseen / len(candidates_set), 6) if candidates_set else 0.0,
            "# Candidate occurrences": int(n_candidates),
        })
        prior_clicked_6h.update(click_set)
    six = pd.DataFrame(six_rows)
    six.to_csv(out_dir / "A3_6hour_eda.csv", index=False, encoding="utf-8-sig")
    return daily, six


def parse_news_entities(row) -> Dict[str, float]:
    entities: Dict[str, float] = {}
    for col in ("title_entities", "abstract_entities"):
        for eid, conf in parse_entities(getattr(row, col)):
            entities[eid] = max(entities.get(eid, 0.0), conf)
    return entities


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            n += 1
    return n


def limit_top10_per_source(counts: Counter) -> List[dict]:
    by_src: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (src, dst), weight in counts.items():
        if src != dst:
            by_src[src].append((dst, int(weight)))
    rows = []
    for src, neighbors in by_src.items():
        neighbors.sort(key=lambda x: (-x[1], x[0]))
        for dst, weight in neighbors[:10]:
            rows.append({"src_type": "news", "src_id": src, "relation": "co_click_next",
                         "dst_type": "news", "dst_id": dst, "weight": weight})
    return rows


def build_relations_and_graphs(beh_train: pd.DataFrame, beh_dev: pd.DataFrame,
                               news: pd.DataFrame, out_dir: Path) -> Dict[str, int]:
    rel_dir = out_dir / "relations"
    graph_dir = out_dir / "graphs"
    rel_dir.mkdir(parents=True, exist_ok=True)
    graph_dir.mkdir(parents=True, exist_ok=True)

    # News metadata / nodes. Prefer train metadata when duplicated.
    news = news.drop_duplicates("news_id", keep="first").copy()
    news_ids = set(news["news_id"].astype(str))
    node_rows: List[dict] = []
    all_user_ids = set(beh_train["user_id"].astype(str)) | set(beh_dev["user_id"].astype(str))
    node_rows.extend({"node_type": "user", "node_id": uid, "label": uid, "category": ""} for uid in sorted(all_user_ids))
    for row in news.itertuples(index=False):
        nid = str(row.news_id)
        node_rows.append({"node_type": "news", "node_id": nid, "label": str(row.title)[:250], "category": str(row.category)})
    mention_edges = []
    news_entities: Dict[str, List[str]] = {}
    entity_ids: Set[str] = set()
    r5_counts: Counter = Counter()
    r3_in = []
    r3_sub_of_pairs: Set[Tuple[str, str]] = set()
    subcat_nodes: Dict[str, Tuple[str, str]] = {}
    category_nodes: Set[str] = set()
    for row in news.itertuples(index=False):
        nid = str(row.news_id)
        entities = parse_news_entities(row)
        unique_entities = sorted(entities.keys())
        news_entities[nid] = unique_entities
        entity_ids.update(unique_entities)
        for eid, confidence in entities.items():
            mention_edges.append({"src_type": "news", "src_id": nid, "relation": "mentions",
                                  "dst_type": "entity", "dst_id": eid, "weight": confidence})
        for i in range(len(unique_entities)):
            for j in range(i + 1, len(unique_entities)):
                r5_counts[(unique_entities[i], unique_entities[j])] += 1
        category = str(row.category).strip()
        subcategory = str(row.subcategory).strip()
        if category:
            category_nodes.add(category)
        if subcategory:
            # Include category in subcategory key to avoid merging identical labels under different categories.
            sub_id = f"{category}::{subcategory}" if category else subcategory
            subcat_nodes[sub_id] = (category, subcategory)
            r3_in.append({"src_type": "news", "src_id": nid, "relation": "in_subcategory",
                          "dst_type": "subcategory", "dst_id": sub_id, "weight": 1.0})
            if category:
                r3_sub_of_pairs.add((sub_id, category))
    node_rows.extend({"node_type": "entity", "node_id": eid, "label": eid, "category": ""} for eid in sorted(entity_ids))
    node_rows.extend({"node_type": "subcategory", "node_id": sid, "label": label, "category": cat}
                     for sid, (cat, label) in sorted(subcat_nodes.items()))
    node_rows.extend({"node_type": "category", "node_id": cat, "label": cat, "category": cat}
                     for cat in sorted(category_nodes))
    write_csv(out_dir / "nodes.csv", ["node_type", "node_id", "label", "category"], node_rows)

    common_cols = ["src_type", "src_id", "relation", "dst_type", "dst_id", "weight"]
    r2_path = rel_dir / "R2_mentions.csv"
    write_csv(r2_path, common_cols, mention_edges)
    r3in_path = rel_dir / "R3_news_in_subcategory.csv"
    write_csv(r3in_path, common_cols, r3_in)
    r3_sub_of = [{"src_type": "subcategory", "src_id": sub_id, "relation": "sub_of",
                  "dst_type": "category", "dst_id": category, "weight": 1.0}
                 for sub_id, category in sorted(r3_sub_of_pairs)]
    r3sub_path = rel_dir / "R3_subcategory_sub_of_category.csv"
    write_csv(r3sub_path, common_cols, r3_sub_of)
    r5_rows = ({"src_type": "entity", "src_id": a, "relation": "co_occurs",
                "dst_type": "entity", "dst_id": b, "weight": count}
               for (a, b), count in sorted(r5_counts.items()))
    r5_path = rel_dir / "R5_entity_cooccur.csv"
    r5_count = write_csv(r5_path, common_cols, r5_rows)

    # History seeds: earliest train impression per user only. Dev is not used as graph history seed.
    hist_sequences, hist_r1_pairs = get_history_seed(beh_train)
    r1_hist = [{"src_type": "user", "src_id": u, "relation": "clicked",
                "dst_type": "news", "dst_id": n, "weight": 1.0} for u, n in sorted(hist_r1_pairs)]
    write_csv(rel_dir / "R1_clicked_Ghist.csv", common_cols, r1_hist)
    r4_hist_counts: Counter = Counter()
    for seq in hist_sequences.values():
        for a, b in zip(seq, seq[1:]):
            if a != b:
                r4_hist_counts[(a, b)] += 1
    r4_hist = limit_top10_per_source(r4_hist_counts)
    write_csv(rel_dir / "R4_co_click_Ghist.csv", common_cols, r4_hist)

    # Parsed positive impression clicks from train rows only. Dates are inclusive via cutoff.
    train_valid = beh_train[beh_train["datetime"].notna()].sort_values(["datetime", "impression_id"]).copy()
    clicks_by_cutoff: Dict[str, Dict[str, Set[str]]] = {}
    for cutoff, date_end in [("0913", "2019-11-13"), ("0914", "2019-11-14")]:
        r1_pairs = set(hist_r1_pairs)
        for row in train_valid[train_valid["date"] <= date_end].itertuples(index=False):
            uid = str(getattr(row, "user_id"))
            for nid, label in parse_impressions(getattr(row, "impressions")):
                if label == 1:
                    r1_pairs.add((uid, nid))
        r1_rows = [{"src_type": "user", "src_id": u, "relation": "clicked",
                    "dst_type": "news", "dst_id": n, "weight": 1.0} for u, n in sorted(r1_pairs)]
        write_csv(rel_dir / f"R1_clicked_Ghist_click_{cutoff}.csv", common_cols, r1_rows)
        clicks_by_cutoff[cutoff] = {"pairs": r1_pairs}

        # For R4, append a news click only when an impression has exactly one positive click.
        # Multiple positives in one impression have no known within-impression order, so their transitions are excluded.
        event_seq: Dict[str, List[Tuple[pd.Timestamp, str, str]]] = defaultdict(list)
        for uid, seq in hist_sequences.items():
            for i, nid in enumerate(seq):
                # Initial-history sequence order is known but exact historical timestamps are not.
                event_seq[uid].append((pd.Timestamp("1900-01-01") + pd.Timedelta(seconds=i), f"hist_{i:08d}", nid))
        ambiguous_multi_click_impressions = 0
        subset = train_valid[train_valid["date"] <= date_end]
        for row in subset.itertuples(index=False):
            positives = [nid for nid, label in parse_impressions(getattr(row, "impressions")) if label == 1]
            # R1 still incorporates all positives; this block only creates ordered R4 transitions.
            if len(positives) == 1:
                event_seq[str(getattr(row, "user_id"))].append((getattr(row, "datetime"), str(getattr(row, "impression_id")), positives[0]))
            elif len(positives) > 1:
                ambiguous_multi_click_impressions += 1
        r4_counts: Counter = Counter()
        for uid, items in event_seq.items():
            # Historical sequence is forced first; impression events sorted by time after it.
            hist_items = [x for x in items if x[1].startswith("hist_")]
            click_items = sorted([x for x in items if not x[1].startswith("hist_")], key=lambda x: (x[0], x[1]))
            ordered = [x[2] for x in hist_items] + [x[2] for x in click_items]
            for a, b in zip(ordered, ordered[1:]):
                if a != b:
                    r4_counts[(a, b)] += 1
        r4_rows = limit_top10_per_source(r4_counts)
        write_csv(rel_dir / f"R4_co_click_Ghist_click_{cutoff}.csv", common_cols, r4_rows)
        print(f"[INFO] R4 cutoff {cutoff}: bỏ qua thứ tự trong {ambiguous_multi_click_impressions:,} impression có >1 click (không có timestamp thứ tự click trong impression).")

    # Common content edges are shared; assemble three snapshot edge lists.
    common_relation_paths = [r2_path, r3in_path, r3sub_path, r5_path]
    dynamic_paths = {
        "Ghist": [rel_dir / "R1_clicked_Ghist.csv", rel_dir / "R4_co_click_Ghist.csv"],
        "Ghist_click_0913": [rel_dir / "R1_clicked_Ghist_click_0913.csv", rel_dir / "R4_co_click_Ghist_click_0913.csv"],
        "Ghist_click_0914": [rel_dir / "R1_clicked_Ghist_click_0914.csv", rel_dir / "R4_co_click_Ghist_click_0914.csv"],
    }
    graph_edge_counts = {}
    for graph_name, dyn_paths in dynamic_paths.items():
        dst = graph_dir / f"{graph_name}_edges.csv"
        with dst.open("w", newline="", encoding="utf-8-sig") as fout:
            writer = csv.DictWriter(fout, fieldnames=common_cols)
            writer.writeheader()
            for path in dyn_paths + common_relation_paths:
                with path.open("r", newline="", encoding="utf-8-sig") as fin:
                    reader = csv.DictReader(fin)
                    for row in reader:
                        writer.writerow(row)
        with dst.open("r", newline="", encoding="utf-8-sig") as fin:
            graph_edge_counts[graph_name] = max(0, sum(1 for _ in fin) - 1)

    # Useful validation file: candidates referenced by behavior logs but absent from merged news metadata.
    behavior_news = set()
    for frame in (beh_train, beh_dev):
        for val in frame["impressions"].tolist():
            behavior_news.update(nid for nid, _ in parse_impressions(val))
        for val in frame["history"].tolist():
            behavior_news.update(parse_space_ids(val))
    missing = sorted(behavior_news - news_ids)
    pd.DataFrame({"news_id": missing}).to_csv(out_dir / "missing_news_ids.csv", index=False, encoding="utf-8-sig")

    manifest = pd.DataFrame([
        {"Graph": "Ghist", "Click label cutoff": "History only; train earliest histories", "Edge file": "graphs/Ghist_edges.csv", "# edges": graph_edge_counts["Ghist"]},
        {"Graph": "Ghist_click_0913", "Click label cutoff": "Train clicks dated 2019-11-09 through 2019-11-13", "Edge file": "graphs/Ghist_click_0913_edges.csv", "# edges": graph_edge_counts["Ghist_click_0913"]},
        {"Graph": "Ghist_click_0914", "Click label cutoff": "Train clicks dated 2019-11-09 through 2019-11-14", "Edge file": "graphs/Ghist_click_0914_edges.csv", "# edges": graph_edge_counts["Ghist_click_0914"]},
    ])
    manifest.to_csv(out_dir / "graph_manifest.csv", index=False, encoding="utf-8-sig")
    return {"R2": len(mention_edges), "R3_news_subcategory": len(r3_in), "R3_subcategory_category": len(r3_sub_of), "R5": r5_count, **graph_edge_counts}


def analyze_graphs(out_dir: Path, skip_community: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute full-graph SNA summary, degree distributions, components; Louvain/NMI by default."""
    try:
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("Cần networkx. Cài bằng: pip install networkx") from exc
    try:
        from sklearn.metrics import normalized_mutual_info_score
    except ImportError:
        normalized_mutual_info_score = None
        skip_community = True
        print("[WARN] Không có scikit-learn; bỏ Louvain/NMI. Cài scikit-learn để tính NMI.")

    graph_dir = out_dir / "graphs"
    node_df = pd.read_csv(out_dir / "nodes.csv", dtype=str, keep_default_na=False)
    news_cat = dict(zip(node_df.loc[node_df.node_type.eq("news"), "node_id"], node_df.loc[node_df.node_type.eq("news"), "category"]))
    summaries = []
    degree_rows = []
    for graph_file in sorted(graph_dir.glob("*_edges.csv")):
        graph_name = graph_file.stem.replace("_edges", "")
        G = nx.Graph()
        relation_counter = Counter()
        # Each edge is read as a row; relation is retained in separate CSV but collapsed in the undirected SNA projection.
        with graph_file.open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                u = (row["src_type"], row["src_id"])
                v = (row["dst_type"], row["dst_id"])
                try:
                    weight = float(row["weight"])
                except (TypeError, ValueError):
                    weight = 1.0
                if u == v:
                    continue
                if G.has_edge(u, v):
                    G[u][v]["weight"] += weight
                else:
                    G.add_edge(u, v, weight=weight)
                relation_counter[row["relation"]] += 1
        n, m = G.number_of_nodes(), G.number_of_edges()
        degree = dict(G.degree())
        degree_by_type: Dict[str, List[int]] = defaultdict(list)
        strength_by_type: Dict[str, List[float]] = defaultdict(list)
        for node, d in degree.items():
            degree_by_type[node[0]].append(int(d))
            strength_by_type[node[0]].append(float(G.degree(node, weight="weight")))
        for node_type, vals in degree_by_type.items():
            freq = Counter(vals)
            for deg, count in sorted(freq.items()):
                degree_rows.append({"Graph": graph_name, "Node type": node_type, "Degree": deg, "Node count": count})
        comp_count = 0
        largest_comp = 0
        comp_sizes_top10 = []
        if n:
            sizes = sorted((len(c) for c in nx.connected_components(G)), reverse=True)
            comp_count = len(sizes)
            largest_comp = sizes[0]
            comp_sizes_top10 = sizes[:10]
        modularity = None
        nmi = None
        community_count = None
        if not skip_community and n and m:
            print(f"[SNA] Louvain + NMI cho {graph_name}: {n:,} nodes, {m:,} undirected edges ...")
            try:
                communities = nx.community.louvain_communities(G, weight="weight", seed=42)
                community_count = len(communities)
                modularity = float(nx.community.modularity(G, communities, weight="weight"))
                community_of = {}
                for i, community in enumerate(communities):
                    for node in community:
                        community_of[node] = i
                news_nodes = [node for node in G.nodes if node[0] == "news" and node[1] in news_cat and news_cat[node[1]]]
                if news_nodes and normalized_mutual_info_score is not None:
                    y_true = [news_cat[node[1]] for node in news_nodes]
                    y_pred = [community_of.get(node, -1) for node in news_nodes]
                    nmi = float(normalized_mutual_info_score(y_true, y_pred))
            except Exception as exc:
                print(f"[WARN] Không tính được Louvain/NMI cho {graph_name}: {exc}")
        type_counts = Counter(node[0] for node in G.nodes)
        summary = {
            "Graph": graph_name,
            "Nodes": n,
            "Undirected edges (projection)": m,
            "Density": float(nx.density(G)) if n > 1 else 0.0,
            "Average degree": (2.0 * m / n) if n else 0.0,
            "Connected components": comp_count,
            "Largest component nodes": largest_comp,
            "Top-10 component sizes": ";".join(str(x) for x in comp_sizes_top10),
            "Louvain communities": community_count,
            "Louvain modularity": modularity,
            "NMI community-vs-news-category": nmi,
            "Nodes user": type_counts.get("user", 0),
            "Nodes news": type_counts.get("news", 0),
            "Nodes entity": type_counts.get("entity", 0),
            "Nodes subcategory": type_counts.get("subcategory", 0),
            "Nodes category": type_counts.get("category", 0),
            "Relation edge rows": int(sum(relation_counter.values())),
        }
        for rel, count in relation_counter.items():
            summary[f"Edges: {rel}"] = count
        summaries.append(summary)
        # Explicitly free current graph before loading the next snapshot.
        del G
    summary_df = pd.DataFrame(summaries).fillna("")
    summary_df.to_csv(out_dir / "SNA_graph_summary.csv", index=False, encoding="utf-8-sig")
    degree_df = pd.DataFrame(degree_rows)
    degree_df.to_csv(out_dir / "SNA_degree_distribution.csv", index=False, encoding="utf-8-sig")
    return summary_df, degree_df


def download_and_extract(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, url, folder in [
        ("MINDsmall_train.zip", TRAIN_URL, data_dir / "MINDsmall_train"),
        ("MINDsmall_dev.zip", DEV_URL, data_dir / "MINDsmall_dev"),
    ]:
        zip_path = data_dir / filename
        if not zip_path.exists():
            print(f"[DOWNLOAD] {url}")
            urllib.request.urlretrieve(url, zip_path)
        folder.mkdir(parents=True, exist_ok=True)
        # Extract only after checking each member remains under destination.
        with zipfile.ZipFile(zip_path) as zf:
            root = folder.resolve()
            for member in zf.infolist():
                target = (folder / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeError(f"Archive contains unsafe path: {member.filename}")
            zf.extractall(folder)
        print(f"[OK] Đã giải nén {filename} vào {folder}")


def main() -> int:
    parser = argparse.ArgumentParser(description="MIND-small E0 EDA + R1-R5 + 3 graph snapshots")
    parser.add_argument("--data-dir", default="data", help="Folder containing MINDsmall_train/ and MINDsmall_dev/")
    parser.add_argument("--train-dir", default=None, help="Optional explicit train folder")
    parser.add_argument("--dev-dir", default=None, help="Optional explicit dev folder")
    parser.add_argument("--output-dir", default="outputs", help="Where tables, relations and graphs are saved")
    parser.add_argument("--download", action="store_true", help="Download official MIND-small train/dev ZIPs (accept dataset terms first)")
    parser.add_argument("--skip-sna", action="store_true", help="Skip graph SNA summary")
    parser.add_argument("--skip-community", action="store_true", help="Skip Louvain modularity and NMI to reduce runtime/memory")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    if args.download:
        download_and_extract(data_dir)
    train_dir = Path(args.train_dir).resolve() if args.train_dir else data_dir / "MINDsmall_train"
    dev_dir = Path(args.dev_dir).resolve() if args.dev_dir else data_dir / "MINDsmall_dev"
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_beh_path = find_named_file(train_dir, "behaviors.tsv")
    dev_beh_path = find_named_file(dev_dir, "behaviors.tsv")
    train_news_path = find_named_file(train_dir, "news.tsv")
    dev_news_path = find_named_file(dev_dir, "news.tsv")
    print(f"[READ] train behaviors: {train_beh_path}")
    print(f"[READ] dev behaviors:   {dev_beh_path}")
    beh_train = read_behaviors(train_beh_path, "train")
    beh_dev = read_behaviors(dev_beh_path, "dev")
    beh_all = pd.concat([beh_train, beh_dev], ignore_index=True)
    news_train = read_news(train_news_path, "train")
    news_dev = read_news(dev_news_path, "dev")
    news_all = pd.concat([news_train, news_dev], ignore_index=True).drop_duplicates("news_id", keep="first")

    profile_rows = []
    for name, bdf, ndf in [("train", beh_train, news_train), ("dev", beh_dev, news_dev), ("combined", beh_all, news_all)]:
        clicks, candidates = count_clicks_and_candidates(bdf)
        profile_rows.append({
            "Split": name,
            "Behavior rows": int(len(bdf)),
            "Unique users": int(bdf["user_id"].nunique()),
            "Unique news in news.tsv": int(ndf["news_id"].nunique()),
            "Clicked candidates": int(clicks),
            "Candidate occurrences": int(candidates),
            "Min timestamp": str(bdf["datetime"].min()) if bdf["datetime"].notna().any() else "",
            "Max timestamp": str(bdf["datetime"].max()) if bdf["datetime"].notna().any() else "",
        })
    pd.DataFrame(profile_rows).to_csv(out_dir / "dataset_profile.csv", index=False, encoding="utf-8-sig")

    print("[EDA] Tạo Bảng A.2, A.3 và kiểm tra History ...")
    a2 = make_a2(beh_train, beh_dev, out_dir)
    daily, six = make_bucket_eda(beh_all, beh_train, out_dir)
    history_summary = make_history_consistency(beh_all, out_dir)
    print("\n[A.2] Statistics:")
    print(a2.to_string(index=False))
    print("\n[A.3] Daily EDA:")
    print(daily.to_string(index=False))
    print("\n[History consistency]:")
    print(history_summary.to_string(index=False))

    print("\n[GRAPH] Dựng R1-R5 và 3 snapshot ...")
    graph_counts = build_relations_and_graphs(beh_train, beh_dev, news_all, out_dir)
    print("[GRAPH] Files generated; edge row counts:")
    for k, v in graph_counts.items():
        print(f"  {k}: {v:,}")

    if not args.skip_sna:
        print("\n[SNA] Phân tích 3 đồ thị; full Louvain/NMI có thể chậm với đồ thị lớn ...")
        summary, _ = analyze_graphs(out_dir, skip_community=args.skip_community)
        print(summary.to_string(index=False))
    else:
        print("[SNA] Đã bỏ qua theo --skip-sna")

    print(f"\n[DONE] Kết quả lưu tại: {out_dir}")
    print("Các bảng chính: A2_split_statistics.csv, A3_daily_eda.csv, A3_6hour_eda.csv, SNA_graph_summary.csv")
    print("Các graph CSV có thể mở bằng Gephi (Data Laboratory -> Import Spreadsheet) hoặc dùng tiếp trong PyTorch Geometric.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[STOP] Đã dừng bởi người dùng.", file=sys.stderr)
        raise SystemExit(130)
