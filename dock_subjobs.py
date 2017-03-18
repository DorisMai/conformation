import pickle
import multiprocessing as mp 
import grids
import sys
from functools import partial 

job_file = sys.argv[1]
timeout = sys.argv[2]

with open(job_file, "rb") as f:
  dock_jobs = pickle.load(f)

partial_docker = partial(dock, timeout=timeout)
pool = mp.Pool(mp.cpu_count())
pool.map(partial_docker, dock_jobs)
pool.terminate()