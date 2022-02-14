#include <iostream>
#include <vector>
//Header file for agent class

class Agent {

	int state, state_0, action;
	int Sx, Sy;
	int S;
	int A;
	std::vector<<std::vector<double>> Q;
	std::vector<double> V;

public:
	Agent(int Sx, int Sy, int action_space);


};