from docplex.mp.model import Model
import itertools
import time
import os
from utils import *
from utils import extract_job_sequence

# Function to create the job sequencing model
def create_job_sequencing_model(n, m, setup_costs, B, J):
    # Create a new model
    model = Model(name='JobSequencing')

    # Define decision variables
    x = {(i, r): model.binary_var(name=f'JobSequence_{i}_{r}') for i in range(1, n + 1) for r in range(1, n + 1)}
    y = {(j, r): model.binary_var(name=f'ToolInMagazine_{j}_{r}') for j in range(1, m + 1) for r in range(1, n + 1)}
    z = {(j, r): model.binary_var(name=f'SetupIndicator_{j}_{r}') for j in range(1, m + 1) for r in range(1, n + 1)}

    # Define the objective function
    model.minimize(model.sum(setup_costs[j - 1] * z[(j, r)] for j in range(1, m + 1) for r in range(1, n + 1)))

    # Constraints

    # Constraint 1: Each job is processed exactly once
    for i in range(1, n + 1):
        model.add_constraint(model.sum(x[(i, r)] for r in range(1, n + 1)) == 1, ctname=f'Job_{i}_Once')

    # Constraint 2: At any time, only one job is processed
    for r in range(1, n + 1):
        model.add_constraint(model.sum(x[(i, r)] for i in range(1, n + 1)) == 1, ctname=f'OneJob_At_Time_{r}')

    # Constraint 3: captures the setup of operations and ensures that the required tools are in the magazine before processing a job.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for r in range(1, n + 1):
                if j in J[i - 1]:
                    model.add_constraint(x[(i, r)] <= y[(j, r)], ctname=f'Setup_{i}_{j}_{r}')

    # Constraint 4: Magazine capacity
    for r in range(1, n + 1):
        model.add_constraint(model.sum(y[(j, r)] for j in range(1, m + 1)) <= B, ctname=f'Magazine_Capacity_{r}')

    # Constraint 5 on Z
    for j in range(1, m + 1):
        for r in range(1, n + 1):
            model.add_constraint(z[(j, r)] >= y[(j, r)] - y[(j, r - 1)] if r > 1 else z[(j, r)] >= y[(j, r)], ctname=f'Constraint_Z1_{j}_{r}')
            model.add_constraint(z[(j, r)] >= y[(j, r - 1)] - y[(j, r)] if r > 1 else z[(j, r)] >= y[(j, r)], ctname=f'Constraint_Z2_{j}_{r}')

    return model, x, y, z 

# Function to create the relaxed job sequencing model
def create_relaxed_job_sequencing_model(n, m, setup_costs, B, J):
    # Create a new model
    model = Model(name='RelaxedJobSequencing')

    # Define decision variables
    x = {(i, r): model.continuous_var(name=f'JobSequence_{i}_{r}', lb=0, ub=1) for i in range(1, n + 1) for r in range(1, n + 1)}
    y = {(j, r): model.binary_var(name=f'ToolInMagazine_{j}_{r}') for j in range(1, m + 1) for r in range(1, n + 1)}
    z = {(j, r): model.binary_var(name=f'SetupIndicator_{j}_{r}') for j in range(1, m + 1) for r in range(1, n + 1)}

    # Define the objective function
    model.minimize(model.sum(setup_costs[j - 1] * z[(j, r)] for j in range(1, m + 1) for r in range(1, n + 1)))

    # Constraints

    # Constraint 1: Each job is processed at most once
    for i in range(1, n + 1):
        model.add_constraint(model.sum(x[(i, r)] for r in range(1, n + 1)) <= 1, ctname=f'Job_{i}_Once')

    # Constraint 2: At any time, only one job is processed
    for r in range(1, n + 1):
        model.add_constraint(model.sum(x[(i, r)] for i in range(1, n + 1)) == 1, ctname=f'OneJob_At_Time_{r}')

    # Constraint 3: captures the setup of operations and ensures that the required tools are in the magazine before processing a job.
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            for r in range(1, n + 1):
                if j in J[i - 1]:
                    model.add_constraint(x[(i, r)] <= y[(j, r)], ctname=f'Setup_{i}_{j}_{r}')

    # Constraint 4: Magazine capacity
    for r in range(1, n + 1):
        model.add_constraint(model.sum(y[(j, r)] for j in range(1, m + 1)) <= B, ctname=f'Magazine_Capacity_{r}')

    # Constraint 5 on Z
    for j in range(1, m + 1):
        for r in range(1, n + 1):
            model.add_constraint(z[(j, r)] >= y[(j, r)] - y[(j, r - 1)] if r > 1 else z[(j, r)] >= y[(j, r)], ctname=f'Constraint_Z1_{j}_{r}')
            model.add_constraint(z[(j, r)] >= y[(j, r - 1)] - y[(j, r)] if r > 1 else z[(j, r)] >= y[(j, r)], ctname=f'Constraint_Z2_{j}_{r}')

    return model, x, y, z

# Function to implement the branch-and-bound algorithm
def branch_and_bound(n, m, setup_costs, B, J):
    best_solution = None
    best_objective_value = float('inf')

    initial_model, x, y, z = create_job_sequencing_model(n, m, setup_costs, B, J)

    stack = [initial_model]

    while stack:
        current_model = stack.pop(0)

        solution = current_model.solve()
        if solution is None:
            continue  # Infeasible solution, backtrack

        objective_value = solution.get_objective_value()

        if objective_value >= best_objective_value:
            continue  # Solution is worse than the best found so far, backtrack

        # Check if the current model has branching variables
        branching_var = None
        branching_value = None

        for var in current_model.iter_binary_vars():
            value = solution.get_value(var)
            if 0 < value < 1:
                if branching_var is None or abs(value - 0.5) > abs(branching_value - 0.5):
                    branching_var = var
                    branching_value = value

        if branching_var is not None:
            # Create two child models by cloning the current model
            child_model_0 = current_model.copy()
            child_model_1 = current_model.copy()

            # Fix the selected branching variable to 0 in one child and 1 in the other
            child_model_0.add_constraint(branching_var == 0)
            child_model_1.add_constraint(branching_var == 1)

            # Add the child models to the stack for further exploration
            stack.append(child_model_0)
            stack.append(child_model_1)
        else:
            # Reached a leaf node with no more branching variables
            if objective_value < best_objective_value:
                best_solution = solution
                best_objective_value = objective_value

    if best_solution is None:
        raise Exception("Infeasible")
    
    if best_solution is not None:
    # Extract the job sequence from the best_solution
        job_sequence = extract_job_sequence(best_solution, n, x)
    return job_sequence, best_objective_value

# Function to implement brute force job sequencing algorithm
def brute_force_job_sequencing(n, m, setup_costs, B, J):
    best_sequence = None
    best_cost = float('inf')

    # Generate all possible permutations of job sequences
    job_permutations = list(itertools.permutations(range(1, n + 1)))

    for job_sequence in job_permutations:
        magazine_status = [0] * m  # Initialize magazine status
        total_setup_time = 0

        for i in job_sequence:
            for j in J[i - 1]:
                # Check if the tool is already in the magazine
                if magazine_status[j - 1] == 0:
                    # Calculate setup time for this tool
                    setup_time = setup_costs[j - 1]

                    # Check if we need to unload a tool from the magazine
                    if sum(magazine_status) >= B:
                        # Find the first tool to unload that is not in J_i
                        tool_index_to_unload = -1
                        for tool_index in range(m - 1, -1, -1):
                            if magazine_status[tool_index] == 1 and tool_index + 1 not in J[i - 1]:
                                tool_index_to_unload = tool_index
                                break

                        if tool_index_to_unload != -1:
                            # Calculate the unloading setup time
                            setup_time += setup_costs[tool_index_to_unload]
                            magazine_status[tool_index_to_unload] = 0

                    # Load the tool into the magazine
                    magazine_status[j - 1] = 1

                    # Update total setup time
                    total_setup_time += setup_time

        # Compare the total setup time with the best solution found so far
        if total_setup_time < best_cost:
            best_cost = total_setup_time
            best_sequence = list(job_sequence)

    return best_sequence, best_cost

# Function to run the job sequencing model and write results to a file
def run_job_sequencing(instance_size, output_file):
    if os.path.exists(output_file):
        os.remove(output_file)

    n, m, setup_costs, B, J = generate_instance(instance_size)
    model, x, y, z = create_job_sequencing_model(n, m, setup_costs, B, J)

    #Solve the problem using docplex directly
    start_time = time.time()
    solution = model.solve()
    end_time = time.time() - start_time

    with open(output_file, 'w') as file:
        if model.solution:
            file.write(f"Optimal Objective Value: {solution.get_objective_value()}\n")
            file.write("Optimal Job Sequence:")
            file.write(f"{extract_job_sequence(model.solution, n, x)}\n")
            file.write("Time for normal solve: {:.2f} sec\n".format(end_time))

            file.write("\n")
    
            # Solve the problem using my brute-force algorithm
            start_time = time.time()
            brute_best_solution, brute_best_obj_value = brute_force_job_sequencing(n, m, setup_costs, B, J)
            end_time = time.time() - start_time

            file.write("Best Job Sequencing Order with my brute-force algorithm: {}\n".format(brute_best_solution))
            file.write("Best Objective value with my brute-force algorithm: {:.2f}\n".format(brute_best_obj_value))
            file.write("Time for my brute-force algorithm: {:.2f} sec\n".format(end_time))

            file.write("\n")

            # Solve the problem using my branch-and-bound algorithm
            start_time = time.time()
            bb_best_solution, bb_best_obj_value = branch_and_bound(n, m, setup_costs, B, J)
            end_time = time.time() - start_time

            file.write("Best Job Sequencing Order with my B&B: {}\n".format(bb_best_solution))
            file.write("Best Objective value with my B&B: {:.2f}\n".format(bb_best_obj_value))
            file.write("Time for my B&B: {:.2f} sec\n".format(end_time))

            file.write("\n")

        else:
            file.write("No solution found.")