from methods import *
from methods import run_job_sequencing

if __name__ == '__main__':
    # Run the model for different instance sizes and write results to files
    run_job_sequencing("small", "small_instance_results.txt")
    run_job_sequencing("medium", "medium_instance_results.txt")
    run_job_sequencing("large", "large_instance_results.txt")

