from datetime import datetime
import time

def create_job_id_name(prefix="job"):
    """
    Create a unique job ID using the current date and time.

    Args:
        prefix (str): Optional prefix for the job ID (default is "job").

    Returns:
        str: A unique job ID string.
    """
    now = datetime.now()
    job_id = f"{prefix}_{now.strftime('%Y%m%d_%H%M%S')}"
    return job_id



def start_timer():
    return time.perf_counter()

def end_timer(start):
    end = time.perf_counter()
    elapsed = end - start
    print(f"Tempo de treino: {elapsed:.2f} segundos")
    return elapsed