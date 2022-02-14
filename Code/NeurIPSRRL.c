#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

////////// OBSCTACLES //////////////////
#define OBS 0
#define TRI 0
#define TBC 0

///////// REFLEXIVE COMPONENTS ////////

#define ENT 1
#define GAM 0
#define WAL 0
#define COR 0

//////////// SETUP ////////////////////

#define Sx 21
#define Sy Sx
#define S (Sx*Sy)   	// world size
#define A 4        	// number of actions
#define T 10000000	// total epochs
#define TM 10000000

/////// FUNCTIONS /////////
int cheobs(int sk) {
#if OBS
	if (((abs(sk%Sx - 14)<2)&&(abs(sk/Sx - 9)<5))
	|| ((abs(sk%Sx - 5)<2)&&(abs(sk/Sx - 5)<9))
	|| ((abs(sk%Sx - 9)<6)&&(abs(sk/Sx - 12)<2))) return(1);
	else return(0);
#elif TRI
        if ((sk<(sk%Sx + Sx*(sk%Sx))) && (sk%Sx<18) && (sk/Sx>2)) return(1);
        else return(0);
#elif TBC
        if (((sk%Sx<6)&&(sk/Sx<6))
        || ((sk%Sx>14)&&(sk/Sx<6))
        || ((sk/Sx ==6) && ((sk%Sx<8) || (sk%Sx > 12)))) return(1);
        else return(0);
#else
        return(0);
#endif
}

int manhattan(int agent, int goal) {
  int dist;

  dist = abs(agent%Sx - goal%Sx) + abs(agent/Sx - goal/Sx);

  return dist;
}

///////////// CODE ////////////////////////

int main() {
	long t=0L;
	int s,s0,s1,a,a0,a1=4,ar,u;
	double eta0=0.1,eta,gamma=0.9,eps=0.75;
	double mmu;
	double rr, tr, taskr;
  double ent, qq;
	double V[S],qn,qa[S][A], qx[A];
  double prior[S][A];
	double mu[S];
	int features[S][A] = {0};
  int goal = 221;
  int start;
  int found = 0;
	int cor = 0;
  int test = 0;
  int manhattan_dist;

	FILE *dat,*mud, *disc;

	eta=eta0;

//// Initialise value functions /////////////////
	for (s=0;s<S;s++) {		// init value function
		for (a=0;a<A;a++) {
			if (cheobs(s)==1) qa[s][a]=0.0;
			else qa[s][a]=drand48();
		}
		if (cheobs(s)==1) mu[s]=-1000.0;
		else mu[s]=1.0;
  }

/// Set sensor readings in features array ////////
	for (s=0; s<S; s++){
		if (((s-Sx)<0) || (cheobs(s-Sx)==1)) features[s][0] = 1;
		if (((s%Sx==Sx-1)|| (cheobs(s+1)==1))) features[s][1] = 1;
		if (((s+Sx)>=S) || (cheobs(s+Sx)==1)) features[s][2] = 1;
		if (((s%Sx==0))|| (cheobs(s-1)== 1)) features[s][3] = 1;
	}

	// Checking signals
	// for (a=0; a<A; a++){
	// 	for (s=0; s<S; s++){
	// 		printf("%d ", features[s][a]);
	// 		if (s%Sx == Sx-1) printf("\n");
	// 	}
	// 	printf("\n");
	// }
	// exit(0);
////////////////////////////////////////////////////

	for (t=0;t<T;t++) {		// loop over epochs
		do {
			s0=(int)(drand48()*(double)S);
		} while (cheobs(s0));

		a1=(int)(drand48()*(double)A);
/////////// Start Episode ////////////////////////////////////////
		for (u=0;u<=2*Sx;u++) { // time steps per epoch for training
			//fprintf(dft,"%d %d\n",s0%Sx,s0/Sx);

			a0=a1;    //this action is now to be used
			ar=a0;

			if (drand48()<eps) ar=(int)(drand48()*(double)A);
						// eps-greedy

			if ((s0<0)||(s0>=S)) {printf("check s0\n"); break;}
			s1=s0;
			rr=0.0;
			cor = 0;
			switch(ar) {  // moving given action
				case 0: if (s0%Sx>0) s1=s0-1;
					break;
				case 1: if (s0/Sx<Sx-1) s1=s0+Sx;
					break;
				case 2:	if (s0%Sx<Sx-1) s1=s0+1;
					break;
				case 3:	if (s0/Sx>0) s1=s0-Sx;
					break;
			}

			mu[s0]+=1.0;

			if (cheobs(s1)) s1=s0;

///////// Reflexive Components Rewards ///////////////
//////// Entropy /////////////////////////////////////
#if ENT
      if (s1==s0) rr-=1.0;

      qq=0.0;
      for (a=0;a<A;a++) {
        qx[a]=exp(qa[s1][a]*5.1);
        qq+=qx[a];
      }
      ent=0.0;
      for (a=0;a<A;a++) {
        ent-=(qx[a]/qq)*log(qx[a]/qq);
      }
      rr+=ent;
#endif
////// Gamma Entropy /////////////////////////////////
#if GAM
			if (s1==s0) rr=-1.0;
#endif
////// Wall favouring /////////////////////////////////
#if WAL
      for (a=0; a<A; a++){
        if (features[s1][a] ==1) cor+=1;
      }
      if (cor>0) rr = 1.0;
#endif
////// Corner favouring //////////////////////////////
#if COR
			for (a=0; a<A; a++){
				if (features[s1][a] ==1) cor+=1;
			}
			if (cor>1) rr = 1.0;
#endif

////// Action Choices /////////////////////////////////
			if ((s1<0)||(s1>=S)) {printf("check s1\n"); break;}

			qn=-1.0e10;
			for (a=0;a<A;a++) {
				if (qa[s1][a]>qn) {
					qn=qa[s1][a];
					a1=a;  // choose new action
				}
			}
			V[s1]=qn;   // value for "next" state

////// Update Value Function /////////////////////////////////

			qa[s0][a0]+=eta*(rr+gamma*V[s1]-qa[s0][a0]);
			s0=s1;
		}

///// End Episode /////////////////////////////////////////////
		if (t%TM==TM-1) { // print out, but not in every step
			//eps=eps0*((double)(T-t)/(double)T)*((double)(T-t)/(double)T);
			//if (eps<0.01) eps=0.01;
			eta=eta0*((double)(T-t)/(double)T)*((double)(T-t)/(double)T);
			printf("%ld %g %g\n",t,eta,eps);

			if ((dat=fopen("dat","wt"))==NULL) printf("file error\n");
			if ((mud=fopen("mud","wt"))==NULL) printf("file error\n");
			for(s=0;s<S;s++) {
				qn=-1.0e10;
				for (a=0;a<A;a++) {
					if (qa[s][a]>qn) {
						qn=qa[s][a];
						a1=a;
					}
				}
				V[s]=qn;
				fprintf(dat,"%d %d %g\n",s%Sx,s/Sx,V[s]);
				if (s%Sx==Sx-1) fprintf(dat,"\n");
			}
			mmu=0.0;
			for(s=0;s<S;s++) {
				mmu+=mu[s];
			}
			mmu/=(double)S;
			for(s=0;s<S;s++) {
				fprintf(mud,"%d %d %g\n",s%Sx,s/Sx,mu[s]/mmu);
				if (s%Sx==Sx-1) fprintf(mud,"\n");
			}
			fclose(dat);
			fclose(mud);
		}
  }
    printf("Testing with task\n");

    for (s=0; s<S; s++){
      for (a=0; a<A; a++){
          prior[s][a] = qa[s][a];
      }
    }
////////// Next Test //////////////////////////////////////////////
// Goal placed at cell 221 (dead centre of square)
  // reduce noise for task search, but non zero
  eps = 0.1;
  for (t=0; t<T; t++){
    found =0;

    do {
      s0=(int)(drand48()*(double)S);
    } while (cheobs(s0));

    start = s0;
    for (u=0; u<2*Sx; u++){

      a0 = a1;
      ar = a0;

      // Choose Action
      if ((s1<0)||(s1>=S)) {printf("check s1\n"); break;}
      s1 = s0;
  		tr=0.0;

      qn=-1.0e10;
      for (a=0;a<A;a++) {
        if (qa[s1][a]>qn) {
          qn=qa[s1][a];
          a1=a;  // choose new action
        }
      }

      switch(ar) {  // moving given action
        case 0: if (s0%Sx>0) s1=s0-1;
        break;
        case 1: if (s0/Sx<Sx-1) s1=s0+Sx;
        break;
        case 2:	if (s0%Sx<Sx-1) s1=s0+1;
        break;
        case 3:	if (s0/Sx>0) s1=s0-Sx;
        break;
      }

      ///// Reward Section /////

      manhattan_dist = manhattan(s1, goal);

      if (manhattan_dist<=2){
        taskr = 1.0/(1.0 + manhattan_dist);
      }
      else{
        taskr = -1.0;
      }
      //////// Entropy /////////////////////////////////////
#if ENT
      if (s1==s0) rr-=1.0;

      qq=0.0;
      for (a=0;a<A;a++) {
        qx[a]=exp(qa[s1][a]*5.1);
        qq+=qx[a];
      }
      ent=0.0;
      for (a=0;a<A;a++) {
        ent-=(qx[a]/qq)*log(qx[a]/qq);
      }
      rr+=ent;
#endif

      tr = taskr + rr;

      ///// New movement ///////////////
      if ((s1<0)||(s1>=S)) {printf("check s1\n"); break;}

      qn=-1.0e10;
      for (a=0;a<A;a++) {
        if (qa[s1][a]>qn) {
          qn=qa[s1][a];
          a1=a;  // choose new action
        }
      }
      V[s1]=qn;   // value for "next" state

      ///// Update Valuation ///////////
      qa[s0][a0]+=eta*(tr+gamma*V[s1]-qa[s0][a0]);
			s0=s1;

      ///// Termination Condition //////
      if (s1 == goal) found =1;
      if (found) break;

    }

    if (found) test++;

    do {
			goal=(int)(drand48()*(double)S);
		} while (cheobs(goal));

    ////// Reset Prior //////////////////
    for (s=0; s<S; s++){
      for (a=0; a<A; a++){
          qa[s][a] = prior[s][a];
      }
    }

  }

  printf("%d/%d\n", test, T);

	fflush(stdout);
	return(1);
}
