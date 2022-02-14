import numpy as np
import matplotlib.pyplot as plt
import argparse
from collections import namedtuple
from tqdm import tqdm
from parfor import pmap

import gridworld as W
import solvers as S
import plot as P
import optimizer as O
import trajectory as T
import maxent as M
import parallel as L

import pickle
import os

world_s='gridworld'
size_s='5'

path = os.getcwd()+"/data/"+world_s+"/"+size_s+"/"
		# open(path+'run.pkl', 'w')
with open(path+'2.0.pkl', 'rb') as filehandle:
	listed = pickle.load(filehandle)

success = 0
fail = 0
for a in listed:
	if a[1] == 1:
		success+=1
	else:
		fail += 1

print(success, fail)

print(listed[0][3].reshape((5,5))/listed[0][3][24])