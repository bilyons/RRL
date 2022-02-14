#include <iostream>
#include <vector>
#include <cmath>
#include "agent.hpp"

///////// ENVIRONMENT /////////////////

#define Sx 11
#define Sy 21
#define S (Sx*Sy)
#define E 10000000
#define TM 10000000

//////// CODE ////////////////////////

int main() {


	int t, e, s;

	// Create agent:
	// Must receive world size, height, and width
	// Must be able to set a location
	// Must be able to retrieve movement and location for obstacle check
	// Public: state
	// Private: action, value function, action-value function

	// Initialise Agents


	for (e=0; e<E; e++){
		//Give agent start state

		for (t=0; t<2*Sx; t++){
			//Select action

			//Move

			std::cout << "hello\n";

			//Check collisions and terminal state

			//Reward

			//Update values

			//End ep

		}


	}


	return(1);
}