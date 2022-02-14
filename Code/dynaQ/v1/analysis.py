import numpy as np
import argparse
from collections import namedtuple
from tqdm import tqdm
from parfor import pmap

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

import pickle
import os

r_dif = 0.5

path = os.getcwd()+"/data/planning_step_100/r_dif/"+str(r_dif)+"/"
		# open(path+'run.pkl', 'w')
with open(path+'100runs.pkl', 'rb') as filehandle:
	listed = pickle.load(filehandle)

successes = 0
iter_successes = 0

for item in listed:

	if item[0] != 0:
		successes += 1

	iter_successes += item[0]

	if item[3] is not None:
		print(item[3].reshape((5,5)))

print(f"Successful iterations: {successes}, Average number of successes: {iter_successes/10}")

