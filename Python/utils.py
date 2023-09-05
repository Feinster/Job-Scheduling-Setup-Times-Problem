import numpy as np

# Function to generate random data for an instance
def generate_instance(instance_size):
    if instance_size == "small":
        n = 3 # Number of jobs
        m = 5 # Number of different tools
        setup_costs = np.random.randint(1, 20, m) # Tool setup costs
        B = 3 # Tool magazine capacity
        J = [list(np.random.choice(range(1, m), np.random.randint(1, m), replace=False)) for _ in range(n)] # List of operations for each job
    elif instance_size == "medium":
        n = 5
        m = 6
        setup_costs = np.random.randint(1, 20, m)
        B = 5
        J = [list(np.random.choice(range(1, m), np.random.randint(1, m), replace=False)) for _ in range(n)]
    elif instance_size == "large":
        n = 10
        m = 5
        setup_costs = np.random.randint(1, 20, m)
        B = 5
        J = [list(np.random.choice(range(1, m), np.random.randint(1, m), replace=False)) for _ in range(n)]
    else:
        raise ValueError("Invalid instance size")

    return n, m, setup_costs, B, J

def extract_job_sequence(solution, n, x):
    job_sequence = []
    for r in range(1, n + 1):
        job_found = False
        for i in range(1, n + 1):
            if solution.get_value(x[(i, r)]) == 1:
                job_sequence.append(i)
                job_found = True
                break
        if not job_found:
            job_sequence.append(None)
    return job_sequence