#include <iostream>
#include <vector>
#include "agent.hpp"
// Class file for agent class

Agent::Agent(int Sx, int Sy, int action_space) {
	: Sx(Sx), Sy(Sy), A(action_space), S(Sx*Sy), Q{}
}


void Agent::move(int action){

	if ((state_0<0)||(state_0>S)) {
		std::cout << 
	}
	switch(action){
		case 0: if (state_0%)

	}
}