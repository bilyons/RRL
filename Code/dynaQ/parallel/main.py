import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import parallel as L
import plot as P
from parfor import pmap
import pickle
import os

def main():

	difs = [0.5, 0.4, 0.3, 0.2,0.1,0.0]

	for r_dif in difs:

		# Go into parallel
		full_size = 5
		p_slip = 0.0
		prelim = 2
		error = 0.01
		run_length = 50
		averaging = 50


		paralleled = L.Parallel(full_size, p_slip, r_dif, prelim, error, run_length)

		listed = pmap(paralleled.parallel, range(averaging))

		path = os.getcwd()+"/data/planning_step_50/r_dif/"+str(r_dif)+"/"

		with open(path+'50runs_1.pkl', 'wb') as filehandle:
			pickle.dump(listed, filehandle)

if __name__ == '__main__':
	main()