

from mpi4py import MPI

import numpy as np
import json
import os
import subprocess
import sys
sys.path.insert(1,'/home/kalyan/SEMESTER_8/RL/macro-action-rl/HFO')
from hfo import *
from model import make_model, simulate
from es import CMAES
import argparse
import time
from config import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def main():
  print("process", rank, "out of total ", comm.Get_size(), "started")
  if (rank == 0):
    print("master")
  else:
    print("slave")

def mpi_fork(n):
  """Re-launches the current script with workers
  Returns "parent" for original parent, "child" for MPI children
  (from https://github.com/garymcintire/mpi_util/)
  """
  if n<=1:
    return "child"
  if os.getenv("IN_MPI") is None:
    env = os.environ.copy()
    env.update(
      MKL_NUM_THREADS="1",
      OMP_NUM_THREADS="1",
      IN_MPI="1"
    )
    print( ["mpirun", "-np", str(n), sys.executable] + sys.argv)
    subprocess.check_call(["mpirun", "-np", str(n), sys.executable] +['-u']+ sys.argv, env=env)
    return "parent"
  else:
    global nworkers, rank
    nworkers = comm.Get_size()
    rank = comm.Get_rank()
    print('assigning the rank and nworkers',rank, nworkers)
    return "child"


if __name__ == "__main__":
  if "parent" == mpi_fork(5): sys.exit()
  main()
