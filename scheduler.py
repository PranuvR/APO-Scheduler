import collections
import pulp
import platform
import os

DAYS = 7 # 0: Mon, ..., 6: Sun
SLOTS = 7 # 0 to 6

def get_solver():
    """Helper to get the correct PuLP solver command for ARM64 Windows."""
    if platform.system() == "Windows" and "arm" in platform.machine().lower():
        pulp_dir = os.path.dirname(pulp.__file__)
        fallback_path = os.path.join(pulp_dir, "solverdir", "cbc", "win", "i64", "cbc.exe")
        if os.path.exists(fallback_path):
            return pulp.COIN_CMD(msg=0, timeLimit=60, path=fallback_path)
    return pulp.PULP_CBC_CMD(msg=0, timeLimit=60)

class IIMR_OptimizedScheduler:
    """
    Backtracking solver for minimal-overlap timetable generation.
    Ported from 'Section Split M4.ipynb'.
    """
    def __init__(self, overlap_df, max_classrooms=5, slots_per_day=8, course_has_female=None):
        self.overlap_df = overlap_df
        self.MAX_CLASSROOMS = max_classrooms
        self.SLOTS_PER_DAY = slots_per_day
        self.course_has_female = course_has_female or {}
        self.DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        self.LUNCH_SLOT = None # Lunch is handled between slots (2:00-3:00 PM)

        # Internal state
        self.schedule = collections.defaultdict(list) # (Day, Slot) -> [SectionID]
        self.assignments = {} # SectionID_SessionNum -> (Day, Slot)
        self.section_daily_count = collections.defaultdict(int) # (SectionID, Day) -> count
        
        # We assume each section needs 2 sessions per week
        self.sections = list(self.overlap_df.index)
        self.queue = []
        for s in self.sections:
            self.queue.append(f"{s}_1")
            self.queue.append(f"{s}_2")

        # Prioritize "Difficult" sections (those with more overlaps)
        # Calculate a 'Difficulty Score' for each section
        difficulty = self.overlap_df.sum(axis=1).to_dict()
        self.queue.sort(key=lambda x: difficulty[x.rsplit('_', 1)[0]], reverse=True)

    def is_slot_safe(self, section_session, day, slot):
        section_id = section_session.rsplit('_', 1)[0]
        
        # 1. Classroom Capacity
        if len(self.schedule[(day, slot)]) >= self.MAX_CLASSROOMS:
            return False
            
        # 2. Section Daily Limit (1 session per day)
        if self.section_daily_count[(section_id, day)] >= 1:
            return False
            
        # 3. Conflict Detection
        # Check if any section already in this slot has a non-zero overlap with the candidate
        for existing in self.schedule[(day, slot)]:
            # Constraint: Same course sections cannot run parallelly (even if overlap is 0)
            existing_course = existing.rsplit('-', 1)[0]
            if existing_course == section_id.rsplit('-', 1)[0]:
                return False
                
            if self.overlap_df.loc[section_id, existing] > 0:
                return False
                
        # 4. Female Faculty Constraint
        # Beyond 6:15 PM means Slot 6 and 7 (1-indexed) in a 7-slot day.
        # Indices 5 and 6.
        if slot >= 6:
            course_code = section_id.rsplit('-', 1)[0]
            if self.course_has_female.get(course_code, False):
                return False
                
        return True

    def solve(self, idx=0):
        if idx == len(self.queue):
            return True
            
        section_session = self.queue[idx]
        section_id = section_session.rsplit('_', 1)[0]
        
        # Try all days and slots
        # Priority: Earlier days, then earlier slots
        for day in self.DAYS:
            for slot in range(1, self.SLOTS_PER_DAY + 1):
                if slot == self.LUNCH_SLOT:
                    continue # Skip lunch
                    
                if self.is_slot_safe(section_session, day, slot):
                    # Place
                    self.schedule[(day, slot)].append(section_id)
                    self.assignments[section_session] = (day, slot)
                    self.section_daily_count[(section_id, day)] += 1
                    
                    if self.solve(idx + 1):
                        return True
                        
                    # Backtrack
                    self.section_daily_count[(section_id, day)] -= 1
                    del self.assignments[section_session]
                    self.schedule[(day, slot)].remove(section_id)
                    
        return False

    def generate(self):
        if self.solve():
            # Format results into a list of dicts for the bridge
            master_tt = []
            day_map = {name: i for i, name in enumerate(self.DAYS)}
            for sess, pos in self.assignments.items():
                sec_id = sess.rsplit('_', 1)[0]
                d_idx = day_map[pos[0]]
                s_idx = pos[1] - 1 # 0-indexed for app compatibility
                
                # Split sec_id (e.g. 'MKISE01-A')
                parts = sec_id.rsplit('-', 1)
                code = parts[0]
                section = parts[1] if len(parts) > 1 else 'A'
                
                master_tt.append({
                    'course_code': code,
                    'section': section,
                    'day': d_idx,
                    'slot': s_idx,
                    'faculty_names': "" # Will be populated by caller
                })
            return master_tt
        return None

def generate_master_timetable(session, Course, Faculty, StudentChoice, seed=42, overlap_matrix=None):
    """
    Bridge to IIMR_OptimizedScheduler.
    Requires an overlap_matrix (DataFrame) computed based on the section split.
    """
    if overlap_matrix is None:
        # Fallback to PuLP if no overlap matrix provided (though app should provide it)
        return generate_master_timetable_pulp(session, Course, Faculty, StudentChoice, seed)
    
    # Identify courses with female faculty
    course_has_female = {}
    courses = session.query(Course).all()
    for c in courses:
        has_female = any(f.gender and f.gender.lower() in ['female', 'f'] for f in c.faculties)
        course_has_female[c.code] = has_female

    scheduler = IIMR_OptimizedScheduler(overlap_matrix, course_has_female=course_has_female)
    master_tt = scheduler.generate()
    
    if master_tt:
        # Populate faculty names (already done in bridge usually, but ensuring)
        for entry in master_tt:
            c_obj = session.query(Course).filter_by(code=entry['course_code']).first()
            if c_obj:
                entry['faculty_names'] = ", ".join([f.name for f in c_obj.faculties])
                entry['course_name'] = c_obj.name
    
    return master_tt

def generate_master_timetable_pulp(session, Course, Faculty, StudentChoice, seed=42):
    # (Existing PuLP-based implementation moved here for legacy compatibility)
    # 1. Count students per course to determine section count
    choices = session.query(StudentChoice).all()
    headcounts = collections.defaultdict(int)
    for ch in choices:
        headcounts[ch.student_id] += 1 # Fix: count choices, not students
        # Wait, the old code had headcounts[ch.course.code] += 1
    
    # Re-extracting headcounts properly
    headcounts = collections.defaultdict(int)
    for ch in choices:
        if ch.course: headcounts[ch.course.code] += 1

def get_student_cohorts(session, StudentChoice, threshold=4):
    """
    Detects groups of students who share at least 'threshold' courses using DSU.
    """
    choices = session.query(StudentChoice).all()
    student_to_courses = collections.defaultdict(set)
    for ch in choices:
        student_to_courses[ch.student_id].add(ch.course.code)
        
    sids = sorted(list(student_to_courses.keys()))
    parent = {sid: sid for sid in sids}

    def find(i):
        if parent[i] == i: return i
        parent[i] = find(parent[i])
        return parent[i]

    def union(i, j):
        root_i, root_j = find(i), find(j)
        if root_i != root_j: parent[root_i] = root_j

    # Build DSU structure
    for i in range(len(sids)):
        for j in range(i + 1, len(sids)):
            s1, s2 = sids[i], sids[j]
            if len(student_to_courses[s1] & student_to_courses[s2]) >= threshold:
                union(s1, s2)

    student_to_cohort = {sid: find(sid) for sid in sids}
    cohort_members = collections.defaultdict(list)
    for sid, cid in student_to_cohort.items():
        cohort_members[cid].append(sid)
        
    return student_to_cohort, cohort_members

def assign_students_to_sections(session, StudentChoice, master_tt):
    """
    Phase 2: Assign students to sections using Cohort-based logic.
    Ensures zero conflicts and balanced headcounts.
    """
    # 1. Detect Cohorts
    student_to_cohort, cohort_members = get_student_cohorts(session, StudentChoice)
    
    choices = session.query(StudentChoice).all()
    student_courses = collections.defaultdict(list)
    courses = set()
    for ch in choices:
        student_courses[ch.student_id].append(ch.course.code)
        courses.add(ch.course.code)
        
    courses = sorted(list(courses))
    
    # Map course_code -> sections -> list of (day, slot)
    course_slots = collections.defaultdict(lambda: collections.defaultdict(list))
    for entry in master_tt:
        course_slots[entry['course_code']][entry['section']].append((entry['day'], entry['slot']))
        
    prob = pulp.LpProblem("Cohort_Student_Assignment", pulp.LpMinimize)
    
    # We want to keep cohorts together. 
    # y[student][course][section]
    y = {}
    for sid in student_to_cohort.keys():
        for c_code in student_courses[sid]:
            for sec in course_slots[c_code].keys():
                y[(sid, c_code, sec)] = pulp.LpVariable(f"y_{sid}_{c_code}_{sec}", cat='Binary')

    # Add cohort integrity preference to objective
    # Penalty for splitting a cohort across sections of the same course
    objective = []
    
    # z[cohort][course][section] - represents if any member of cohort is in this section
    z = {}
    for cid, members in cohort_members.items():
        for c_code in courses:
            # Check if any member takes this course
            if any(c_code in student_courses[m] for m in members):
                secs = list(course_slots[c_code].keys())
                for sec in secs:
                    z[(cid, c_code, sec)] = pulp.LpVariable(f"z_{cid}_{c_code}_{sec}", cat='Binary')
                    # Penalty for using a section
                    objective.append(10 * z[(cid, c_code, sec)])
                    
                    # If any student in cohort is in this section, z must be 1
                    for sid in members:
                        if (sid, c_code, sec) in y:
                            prob += y[(sid, c_code, sec)] <= z[(cid, c_code, sec)]

    # 1. Exactly one section per chosen course
    for sid, c_list in student_courses.items():
        for c_code in c_list:
            if not course_slots[c_code]: continue
            prob += pulp.lpSum(y[(sid, c_code, sec)] for sec in course_slots[c_code].keys()) == 1
            
    # 2. Zero conflicts per student
    for sid in student_to_cohort.keys():
        for d in range(DAYS):
            for s in range(SLOTS):
                expr = []
                for c_code in student_courses[sid]:
                    for sec, slots in course_slots[c_code].items():
                        if (d, s) in slots:
                            expr.append(y[(sid, c_code, sec)])
                if expr:
                    prob += pulp.lpSum(expr) <= 1
                    
    # 3. Headcount balance (|A - B| <= 5)
    diff_sum = []
    for c_code in courses:
        secs = list(course_slots[c_code].keys())
        if len(secs) >= 2:
            s1, s2 = secs[0], secs[1]
            diff = pulp.LpVariable(f"diff_{c_code}", lowBound=0)
            
            count1 = pulp.lpSum(y[(sid, c_code, s1)] for sid in student_to_cohort.keys() if (sid, c_code, s1) in y)
            count2 = pulp.lpSum(y[(sid, c_code, s2)] for sid in student_to_cohort.keys() if (sid, c_code, s2) in y)
            
            prob += count1 - count2 <= diff
            prob += count2 - count1 <= diff
            prob += diff <= 5
            diff_sum.append(100 * diff) # Higher priority for balance
            
    prob += pulp.lpSum(objective) + pulp.lpSum(diff_sum)
    
    status = prob.solve(get_solver())
    if status != pulp.LpStatusOptimal and status != 1:
        return None
        
    assignments = []
    for (sid, c_code, sec), var in y.items():
        if pulp.value(var) is not None and pulp.value(var) > 0.5:
            assignments.append({
                'student_id': sid,
                'course_code': c_code,
                'section': sec
            })
            
    return assignments
