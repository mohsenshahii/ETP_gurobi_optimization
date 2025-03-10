from gurobipy import Model, GRB
import numpy as np

# Load instance01 data
exam_file = "instance02.exm"
slot_file = "instance02.slo"
stu_file = "instance02.stu"

# Read number of students per exam
exams = {}
with open(exam_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:  # Skip empty lines
            eid, num_students = map(int, line.split())
            exams[eid] = num_students

# Read number of available time slots
with open(slot_file, 'r') as f:
    num_slots = int(f.readline().strip())

# Read student enrollments
students = {}
with open(stu_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line:  # Skip empty lines
            sid, eid = line.split()
            sid = int(sid[1:])  # Remove 's' from student ID
            eid = int(eid)
            if sid not in students:
                students[sid] = []
            students[sid].append(eid)
total_students = len(students)

# Create conflict matrix (two exams conflict if they share students)
n_exams = len(exams)
conflicts = np.zeros((n_exams + 1, n_exams + 1), dtype=int)
for s in students.values():
    for i in range(len(s)):
        for j in range(i + 1, len(s)):
            e1, e2 = s[i], s[j]
            conflicts[e1][e2] += 1
            conflicts[e2][e1] += 1

# Define ILP model
model = Model("Examination Timetabling")
model.setParam('OutputFlag', 0)  # Suppress output
model.setParam('TimeLimit', 600)  # Set time limit to 2 minutes

# Decision variables: x[e, t] = 1 if exam e is scheduled in time slot t
x = model.addVars(exams.keys(), range(num_slots), vtype=GRB.BINARY, name="x")

# Constraint: Each exam must be scheduled exactly once
for e in exams.keys():
    model.addConstr(sum(x[e, t] for t in range(num_slots)) == 1)

# Constraint: Conflicting exams cannot be in the same time slot
for e1 in exams.keys():
    for e2 in exams.keys():
        if conflicts[e1][e2] > 0 and e1 < e2:
            for t in range(num_slots):
                model.addConstr(x[e1, t] + x[e2, t] <= 1)

# Objective function: Minimize penalty for closely scheduled conflicting exams
penalty = 0
for e1 in exams.keys():
    for e2 in exams.keys():
        if conflicts[e1][e2] > 0 and e1 < e2:
            for d in range(1, 6):  # Penalize conflicts within 5 slots
                for t in range(num_slots - d):
                    # Create auxiliary binary variable for the product
                    y = model.addVar(vtype=GRB.BINARY, name=f"y_{e1}_{e2}_{t}_{d}")
                    # Add linearization constraints
                    model.addConstr(y <= x[e1, t])
                    model.addConstr(y <= x[e2, t + d])
                    model.addConstr(y >= x[e1, t] + x[e2, t + d] - 1)
                    # Add penalty term
                    penalty += (2 ** (5 - d) * conflicts[e1][e2] / total_students) * y

model.setObjective(penalty, GRB.MINIMIZE)

# Solve model
model.optimize()

# Check if the model is solved to optimality
if model.status == GRB.OPTIMAL:
    # Display solution
    schedule = {}
    for e in exams.keys():
        for t in range(num_slots):
            if x[e, t].X > 0.5:  # Access the value of the variable
                schedule[e] = t

    print("Optimal Exam Schedule:")
    for e, t in sorted(schedule.items()):
        print(f"Exam {e} -> Time Slot {t}")

    # Calculate final penalty
    final_penalty = 0
    for e1 in exams.keys():
        for e2 in exams.keys():
            if conflicts[e1][e2] > 0 and e1 < e2:
                t1 = schedule[e1]
                t2 = schedule[e2]
                d = abs(t1 - t2)
                if 1 <= d <= 5:
                    final_penalty += (2 ** (5 - d) * conflicts[e1][e2] / total_students)

    print(f"\nFinal Penalty Score: {final_penalty:.2f}")
else:
    print("No optimal solution found.") 