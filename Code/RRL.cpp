#include <iostream>
#include <cmath>
#include <cstdlib>
using namespace std;

#define Sx 21
#define Sy Sx
#define S (Sx*Sy)
#define A 4
#define T 10000000
#define TM 10000000

// Agent needs to be able to:
// Move
    // Move requires check obstacle
// Get reward
// Update V[S] and qa[S][A]

class Agent {
public:
  int origin;
  int cur_state, old_state;
  int world_size, action_space=4;
  int world_width, world_height;
  double qa[S][A];
  double rr;

  // Member functions declaration
  void set_origin(int ori);
  void getWorldInfo(int states);
  double getReward (void);

};

void Agent::set_origin (int ori){
  origin = ori;
}

double Agent:: getReward(void){
  return 0; // insert reward function
}

int main(){
  int origin;
  // Creating agents
  Agent Agent1;
  Agent Agent2;

  // Set seed
  // srand((unsigned int)time(NULL));

  //Agent1 specification
  Agent1.set_origin(5);
  origin = Agent1.origin;
  cout << "The random number is: " << origin << endl;
  return 0;

}
