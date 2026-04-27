import json
import os
import re

from catalog import (
    _extract_field_from_course_text,
    _normalize_course_code,
    _semester_sort_index,
    get_course_ects_map,
    get_curriculum_course_positions,
)
from parsing import _dedupe_keep_order, normalize_department_code


def _build_scoped_course_index(dept_code, start_year=None, program_name=None, curriculum_url=None):
    positions = get_curriculum_course_positions(
        dept_code,
        start_year=start_year,
        program_name=program_name,
        curriculum_url=curriculum_url,
    )
    candidate_codes = set(positions.keys())
    if not candidate_codes:
        return {}, positions

    if not os.path.exists("courses.json"):
        return {}, positions

    try:
        with open("courses.json", "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  courses.json read error: {e}")
        return {}, positions

    per_code_candidates = {}
    for item in data:
        dep_code = _normalize_course_code(item.get("code"))
        if not dep_code or dep_code not in candidate_codes:
            continue

        text = item.get("text") or ""
        prereq_lines = re.findall(r"^Prerequisites:\s*(.+)$", text, re.IGNORECASE | re.MULTILINE)
        prereq_text = " ".join(prereq_lines)
        prereq_codes = [_normalize_course_code(x) for x in re.findall(r"\b[A-Z]{2,6}\s*\d{3,4}\b", prereq_text)]
        prereq_codes = sorted(set([p for p in prereq_codes if p]))

        item_dept = normalize_department_code((item.get("department") or "").strip().upper())
        item_program = (item.get("program_name") or "").strip()
        semester_raw = (_extract_field_from_course_text(text, "Semester") or "").strip().lower()
        offered_term = None
        if semester_raw.startswith("fall"):
            offered_term = "Fall"
        elif semester_raw.startswith("spring"):
            offered_term = "Spring"
        course_type = (_extract_field_from_course_text(text, "Course Type") or "").lower()
        target_term = (positions.get(dep_code, {}).get("term") or "").lower()

        score = 0
        if item_dept == dept_code:
            score += 100
        elif item_dept:
            score -= 100
        if program_name:
            if item_program and item_program.lower().strip() == program_name.lower().strip():
                score += 100
            elif item_program:
                score -= 25
        if target_term and target_term in semester_raw:
            score += 10
        if "service" in course_type:
            score -= 1

        per_code_candidates.setdefault(dep_code, []).append({
            "score": score,
            "prereqs": prereq_codes,
            "offered_term": offered_term,
        })

    scoped_index = {}
    for dep_code, candidates in per_code_candidates.items():
        if not candidates:
            continue
        candidates.sort(key=lambda x: (-x["score"], len(x["prereqs"])))
        best = candidates[0]
        scoped_index[dep_code] = {
            "prereqs": set(best["prereqs"]),
            "offered_term": best.get("offered_term") or positions.get(dep_code, {}).get("term"),
        }

    # Ensure every curriculum code has at least term metadata even if
    # the detailed catalog record is unavailable.
    for code, pos in positions.items():
        scoped_index.setdefault(code, {
            "prereqs": set(),
            "offered_term": pos.get("term"),
        })

    return scoped_index, positions


def get_scoped_prerequisite_maps(dept_code, start_year=None, program_name=None, curriculum_url=None):
    scoped_index, _ = _build_scoped_course_index(
        dept_code,
        start_year=start_year,
        program_name=program_name,
        curriculum_url=curriculum_url,
    )
    prereq_map = {}
    for dep_code, meta in scoped_index.items():
        prereq_map[dep_code] = set(meta.get("prereqs") or set())

    reverse_graph = {}
    for dep_code, prereqs in prereq_map.items():
        for prereq in prereqs:
            reverse_graph.setdefault(prereq, set()).add(dep_code)

    return prereq_map, reverse_graph


def get_scoped_offered_terms(dept_code, start_year=None, program_name=None, curriculum_url=None):
    scoped_index, _ = _build_scoped_course_index(
        dept_code,
        start_year=start_year,
        program_name=program_name,
        curriculum_url=curriculum_url,
    )
    return {
        code: meta.get("offered_term")
        for code, meta in scoped_index.items()
        if meta.get("offered_term")
    }


def _index_to_year_term(index):
    idx = int(index)
    year = idx // 2
    term = "Fall" if idx % 2 == 0 else "Spring"
    return year, term


def _align_index_to_offering(index, offered_term):
    term = (offered_term or "").strip().lower()
    idx = int(index)
    if term.startswith("fall") and idx % 2 == 1:
        return idx + 1
    if term.startswith("spring") and idx % 2 == 0:
        return idx + 1
    return idx


def build_ects_aware_recovery_schedule(recovery_order, gpa_value, current_index, prereq_map=None, offered_term_by_code=None):
    if not recovery_order:
        return [], 32 if gpa_value is None else (44 if gpa_value >= 2.5 else 38 if gpa_value >= 2.0 else 32)

    ects_map = get_course_ects_map()
    max_ects = 32 if gpa_value is None else (44 if gpa_value >= 2.5 else 38 if gpa_value >= 2.0 else 32)
    prereq_map = prereq_map or {}
    offered_term_by_code = offered_term_by_code or {}
    recovery_codes = {item.get("code") for item in recovery_order if item.get("code")}
    planned_index_by_code = {
        item.get("code"): int(item.get("index", current_index))
        for item in recovery_order
        if item.get("code")
    }

    slots = {}
    scheduled_index = {}
    for item in recovery_order:
        code = item.get("code")
        if not code:
            continue
        ects = ects_map.get(code, 0.0)
        target_index = item.get("index", current_index)
        if target_index < current_index:
            target_index = current_index
        offered_term = item.get("term") or offered_term_by_code.get(code)
        target_index = _align_index_to_offering(target_index, offered_term)

        # Strict prerequisite sequencing: if prerequisite is in the recovery set,
        # this course cannot be in the same semester as that prerequisite.
        for prereq in prereq_map.get(code, set()):
            if prereq not in recovery_codes:
                continue
            if prereq in scheduled_index:
                target_index = max(target_index, scheduled_index[prereq] + 1)
            else:
                target_index = max(target_index, planned_index_by_code.get(prereq, current_index) + 1)
        target_index = _align_index_to_offering(target_index, offered_term)

        while True:
            used = sum(x.get("ects", 0.0) for x in slots.get(target_index, []))
            if ects <= 0:
                slots.setdefault(target_index, []).append({
                    "code": code,
                    "name": item.get("name", ""),
                    "ects": ects,
                    "reason": item.get("reason", "dependency order"),
                })
                break
            if used + ects <= max_ects:
                slots.setdefault(target_index, []).append({
                    "code": code,
                    "name": item.get("name", ""),
                    "ects": ects,
                    "reason": item.get("reason", "dependency order"),
                })
                break
            if (offered_term or "").strip().lower() in ["fall", "spring"]:
                target_index += 2
            else:
                target_index += 1

        scheduled_index[code] = target_index

    scheduled = []
    for idx in sorted(slots.keys()):
        year, term = _index_to_year_term(idx)
        items = slots[idx]
        total = sum(x.get("ects", 0.0) for x in items)
        scheduled.append({
            "index": idx,
            "year": year,
            "term": term,
            "total_ects": total,
            "items": items,
        })

    return scheduled, max_ects


def get_blocked_dependency_chain(
    failed_courses,
    dept_code,
    year,
    term,
    start_year=None,
    program_name=None,
    curriculum_url=None,
    reverse_graph=None,
    include_past=True,
):
    failed = _dedupe_keep_order([_normalize_course_code(c) for c in (failed_courses or []) if _normalize_course_code(c)])
    if not failed:
        return []

    positions = get_curriculum_course_positions(
        dept_code,
        start_year=start_year,
        program_name=program_name,
        curriculum_url=curriculum_url,
    )
    if not positions:
        return []

    reverse_graph = reverse_graph or get_scoped_prerequisite_maps(
        dept_code,
        start_year=start_year,
        program_name=program_name,
        curriculum_url=curriculum_url,
    )[1]
    if not reverse_graph:
        return []

    try:
        current_idx = _semester_sort_index(int(year), term)
    except Exception:
        return []

    seen = set(failed)
    depth = {c: 0 for c in failed}
    parent = {}
    frontier = list(failed)

    while frontier:
        next_frontier = []
        for prereq in frontier:
            for dep in sorted(reverse_graph.get(prereq, [])):
                if dep in seen:
                    continue
                pos = positions.get(dep)
                if not pos:
                    continue
                if (not include_past) and pos["index"] < current_idx:
                    continue
                seen.add(dep)
                depth[dep] = depth.get(prereq, 0) + 1
                parent[dep] = prereq
                next_frontier.append(dep)
        frontier = next_frontier

    blocked = []
    for code, blocker in parent.items():
        pos = positions.get(code)
        if not pos:
            continue
        blocked.append({
            "code": code,
            "name": pos.get("name", ""),
            "year": pos.get("year"),
            "term": pos.get("term"),
            "blocked_by": blocker,
            "depth": depth.get(code, 1),
            "index": pos.get("index", 9999),
        })

    blocked.sort(key=lambda x: (x["index"], x["depth"], x["code"]))
    return blocked


def build_suggested_recovery_order(
    failed_courses,
    blocked_chain,
    dept_code,
    start_year=None,
    program_name=None,
    curriculum_url=None,
):
    failed = _dedupe_keep_order([_normalize_course_code(c) for c in (failed_courses or []) if _normalize_course_code(c)])
    positions = get_curriculum_course_positions(
        dept_code,
        start_year=start_year,
        program_name=program_name,
        curriculum_url=curriculum_url,
    )

    order_items = []
    for code in failed:
        pos = positions.get(code, {})
        order_items.append({
            "code": code,
            "name": pos.get("name", ""),
            "year": pos.get("year"),
            "term": pos.get("term"),
            "depth": 0,
            "index": pos.get("index", 0),
            "reason": "failed course",
        })

    for item in blocked_chain or []:
        order_items.append({
            "code": item.get("code"),
            "name": item.get("name"),
            "year": item.get("year"),
            "term": item.get("term"),
            "depth": item.get("depth", 1),
            "index": item.get("index", 9999),
            "reason": f"depends on {item.get('blocked_by')}",
        })

    dedup = {}
    for item in order_items:
        code = item.get("code")
        if not code:
            continue
        if code not in dedup:
            dedup[code] = item
            continue
        prev = dedup[code]
        if (item.get("depth", 99), item.get("index", 9999)) < (prev.get("depth", 99), prev.get("index", 9999)):
            dedup[code] = item

    ordered = sorted(
        dedup.values(),
        key=lambda x: (x.get("depth", 99), x.get("index", 9999), x.get("code", "")),
    )
    return ordered


def _safe_alternative_courses_for_term(
    dept,
    year,
    term,
    current_codes,
    blocked_codes,
    positions,
    prereq_map,
    limit=8,
):
    try:
        current_idx = _semester_sort_index(int(year), term)
    except Exception:
        return []

    want_term = (term or "").strip().lower()
    alternatives = []

    for code, pos in positions.items():
        if code.startswith("ELEC"):
            continue
        if code in current_codes or code in blocked_codes:
            continue
        idx = pos.get("index", -1)
        if idx <= current_idx:
            continue
        if (pos.get("term") or "").strip().lower() != want_term:
            continue

        prereqs = prereq_map.get(code, set())
        if any(p in blocked_codes for p in prereqs):
            continue

        alternatives.append({
            "code": code,
            "name": pos.get("name", ""),
            "year": pos.get("year"),
            "term": pos.get("term"),
            "index": idx,
        })

    alternatives.sort(key=lambda x: (x["index"], x["code"]))
    return alternatives[:limit]
