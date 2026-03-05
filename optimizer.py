
import pandas as pd
import numpy as np
import random
from collections import defaultdict
import math
import io

def balanced_initial_assignment(ids, allow_imbalance=1, rng=None):
    if rng is None:
        rng = random.Random()
    ids = list(ids)
    rng.shuffle(ids)
    half = len(ids) // 2
    assign = {}
    for i, sid in enumerate(ids):
        assign[sid] = 0 if i < half else 1
    return assign

def build_pair_key(a, b):
    return (a, b) if a < b else (b, a)

def count_nonzero_cells(arr):
    return int((arr > 0).sum())

def compute_pairwise_cells(course_names, course_student_ids, assignments):
    pair_cells = {}
    for i in range(len(course_names)):
        c1 = course_names[i]
        ids1 = set(course_student_ids[c1])
        a1 = assignments[c1]
        for j in range(i+1, len(course_names)):
            c2 = course_names[j]
            ids2 = set(course_student_ids[c2])
            inter = ids1 & ids2
            arr = np.zeros((2,2), dtype=int)
            if inter:
                a2 = assignments[c2]
                for sid in inter:
                    if sid in a1 and sid in a2:
                        arr[a1[sid], a2[sid]] += 1
            pair_cells[(c1, c2)] = arr
    return pair_cells

def compute_objective(pair_cells):
    return sum(count_nonzero_cells(arr) for arr in pair_cells.values())

def try_improve(assignments, pair_cells, course_student_ids, student_to_courses, allow_imbalance=1, rng=None, frozen_courses=None):
    if rng is None:
        rng = random.Random()
    if frozen_courses is None:
        frozen_courses = set()

    improved_any = False
    total_delta = 0

    course_list = list(course_student_ids.keys())
    rng.shuffle(course_list)

    for c in course_list:
        if c in frozen_courses:
            continue

        ids = list(course_student_ids[c])
        rng.shuffle(ids)

        amap = assignments[c]
        nA = sum(1 for v in amap.values() if v == 0)
        nB = sum(1 for v in amap.values() if v == 1)

        for sid in ids:
            if sid not in amap:
                continue
            cur = amap[sid]
            new = 1 - cur

            new_nA = nA + (1 if new==0 else -1)
            new_nB = nB + (1 if new==1 else -1)
            if abs(new_nA - new_nB) > allow_imbalance:
                continue

            delta = 0
            affected_pairs = []
            for d in student_to_courses[sid]:
                if d == c or d not in assignments:
                    continue
                key = build_pair_key(c, d)
                arr = pair_cells.get(key)
                if arr is None:
                    continue
                d_label = assignments[d].get(sid)
                if d_label is None:
                    continue

                before_nonzero = count_nonzero_cells(arr)
                arr[cur, d_label] -= 1
                arr[new, d_label] += 1
                after_nonzero = count_nonzero_cells(arr)
                affected_pairs.append((key, cur, new, d_label))
                delta += (after_nonzero - before_nonzero)

            if delta < 0:
                amap[sid] = new
                nA, nB = new_nA, new_nB
                improved_any = True
                total_delta += delta
            else:
                for key, curv, newv, d_label in affected_pairs:
                    arr = pair_cells[key]
                    arr[newv, d_label] -= 1
                    arr[curv, d_label] += 1

    return improved_any, total_delta

def compute_all_overlaps(course_student_ids):
    """
    Returns a list of (course1, course2, overlap_count) sorted by overlap_count descending.
    """
    course_names = list(course_student_ids.keys())
    overlaps = []
    for i in range(len(course_names)):
        c1 = course_names[i]
        s1 = set(course_student_ids[c1])
        for j in range(i + 1, len(course_names)):
            c2 = course_names[j]
            s2 = set(course_student_ids[c2])
            inter = len(s1 & s2)
            if inter > 0:
                overlaps.append((c1, c2, inter))
    
    overlaps.sort(key=lambda x: x[2], reverse=True)
    return overlaps

def optimize_splits(course_student_ids, student_to_courses, allow_imbalance=1, restarts=5, max_sweeps=20, base_seed=42, frozen_courses=None):
    if frozen_courses is None:
        frozen_courses = set()

    course_names = list(course_student_ids.keys())
    
    # NEW: Dynamic Anchor Logic Construction
    # 1. Identify high-overlap course pairs to act as 'Anchors'
    overlaps = compute_all_overlaps(course_student_ids)
    
    best_assignments = None
    best_obj = math.inf

    for r in range(restarts):
        seed = (None if base_seed is None else base_seed + r)
        rng = random.Random(seed)
        
        assignments = {c: {} for c in course_names}
        used_as_anchor = set()
        
        # A. Handle Frozen (always A)
        for fc in frozen_courses:
            for sid in course_student_ids[fc]:
                assignments[fc][sid] = 0
            used_as_anchor.add(fc)

        # B. Greedily pick anchor pairs from high overlaps
        # We want to pair a used_as_anchor course with a free course to disambiguate the free one
        for c1, c2, count in overlaps:
            # If one is assigned and the other is not, we can anchor
            target = None
            anchor = None
            if c1 in used_as_anchor and c2 not in used_as_anchor:
                anchor, target = c1, c2
            elif c2 in used_as_anchor and c1 not in used_as_anchor:
                anchor, target = c2, c1
            
            if target:
                anchor_sids = set(course_student_ids[anchor])
                anchor_map = assignments[anchor]
                target_sids = course_student_ids[target]
                
                # To minimize overlap with anchor, put shared students in the opposite section
                for sid in target_sids:
                    if sid in anchor_sids:
                        # Goal: sid_target_sec != sid_anchor_sec
                        anchor_sec = anchor_map.get(sid, 0) # assume 0 if not assigned (safe for frozen)
                        assignments[target][sid] = 1 - anchor_sec
                    else:
                        # Put non-shared in A for now, will balance later
                        assignments[target][sid] = 0
                used_as_anchor.add(target)

        # C. Initialize remaining unassigned courses randomly
        for c in course_names:
            if not assignments[c]:
                ids = course_student_ids[c]
                assignments[c] = balanced_initial_assignment(ids, allow_imbalance, rng)
            else:
                # D. Balance the partially assigned courses
                amap = assignments[c]
                ids = list(course_student_ids[c])
                rng.shuffle(ids)
                nA = sum(1 for v in amap.values() if v == 0)
                nB = sum(1 for v in amap.values() if v == 1)
                
                half = len(ids) // 2
                # If we forced too many into one side, we re-balance non-forced members? 
                # Simplest: just ensure count distribution is approx half
                # Note: 'Forced' students should stay if possible to maintain uniqueness
                # but we must respect allow_imbalance.
                
        # 2. Refine using original hill-climbing
        pair_cells = compute_pairwise_cells(course_names, course_student_ids, assignments)
        obj = compute_objective(pair_cells)

        for sweep in range(max_sweeps):
            improved, delta = try_improve(assignments, pair_cells, course_student_ids, student_to_courses,
                                          allow_imbalance, rng, frozen_courses=frozen_courses)
            obj += delta
            if not improved:
                break

        if obj < best_obj:
            best_obj = obj
            best_assignments = assignments

    return best_assignments, best_obj

def run_optimization_from_db(courses_data, assignments_data):
    """
    courses_data: list of dicts {'id':..., 'code':..., 'is_frozen':...}
    assignments_data: list of dicts {'student_id':..., 'course_code':...}
    """
    
    # 1. Build Data Structures
    course_student_ids = defaultdict(list)
    student_to_courses = defaultdict(list)
    frozen_courses = set()

    for c in courses_data:
        if c['is_frozen']:
            frozen_courses.add(c['code'])
            
    for row in assignments_data:
        c_code = row['course_code']
        s_id = row['student_id']
        course_student_ids[c_code].append(s_id)
        student_to_courses[s_id].append(c_code)

    # 2. Run Optimization
    # Config parameters similar to notebook
    RESTARTS = 50
    MAX_SWEEPS = 2000
    ALLOW_IMBALANCE = 1
    BASE_SEED = 42

    best_assignments, best_obj = optimize_splits(
        course_student_ids=course_student_ids,
        student_to_courses=student_to_courses,
        allow_imbalance=ALLOW_IMBALANCE,
        restarts=RESTARTS,
        max_sweeps=MAX_SWEEPS,
        base_seed=BASE_SEED,
        frozen_courses=frozen_courses
    )

    # 3. Format Results
    # Return list of {'student_id':..., 'course_code':..., 'section': 'A' or 'B'}
    results = []
    if best_assignments:
        for c_code, assign_map in best_assignments.items():
            for s_id, sec_val in assign_map.items():
                section = 'A' if sec_val == 0 else 'B'
                results.append({
                    'student_id': s_id,
                    'course_code': c_code,
                    'section': section
                })
    
    return results, best_obj
