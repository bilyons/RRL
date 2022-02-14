import numpy as np
import argparse
from collections import namedtuple
from tqdm import tqdm
from parfor import pmap

np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})

import pickle
import os, os.path


r_dif = 0.3
# path = os.getcwd()+"/data/planning_step_50/r_dif/"+str(r_dif)+"/"
# x = len([name for name in os.listdir(path) if os.path.isfile(name)])

# print(x)
# exit()

path = os.getcwd()+"/data/planning_step_50/r_dif/"+str(r_dif)+"/"
		# open(path+'run.pkl', 'w')
with open(path+'50runs_0.pkl', 'rb') as filehandle:
	listed = pickle.load(filehandle)

tot_suc = 0
avg_suc = 0
avg_rew = np.zeros(25)
avg_error = 0
for r in listed:
	if r[0] > 0:
		tot_suc += 1
	avg_suc += r[0]

	avg_rew += r[3]
	avg_error += r[2]


	print(r)

avg_rew = avg_rew/50
avg_error = avg_error/50
print(avg_rew.reshape((5,5)))
print(avg_error)
print(f"Total successes: {tot_suc}")
print(f"Average successes: {avg_suc/50}")

