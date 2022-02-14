import numpy as np
import argparse
from collections import namedtuple
from tqdm import tqdm
from parfor import pmap
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

import pickle
import os

world_s='gridworld'
size_s='5'

path = os.getcwd()+"/data/"+world_s+"/"+size_s+"/"
		# open(path+'run.pkl', 'w')
with open(path+'0.5.pkl', 'rb') as filehandle:
	listed = pickle.load(filehandle)
# print(listed)

success = 0
fail = 0
suc_avg = 0
for a in listed[0:49]:
	if a[1] == 1:
		success+=1
		suc_avg += a[2]
		print(a[2])
		# print(a[3].reshape((11,11)))
	else:
		fail += 1
		# print(a)

print(success, fail)
# print(suc_avg/success)
# print(listed)
# print(listed[498])

# print(listed[498][3].reshape((5,5))/listed[498][3][24])

# for a in listed:
# 	if a[1] == 0:
# 		print(listed[a][3])