#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

////////// OBSCTACLES //////////////////
#define OBS 0
#define TRI 1
#define TBC 0

///////// REFLEXIVE COMPONENTS ////////

// Single agent variants
#define ENT 0
#define GAM 0
#define WAL 0
#define COR 0

// Multi agent variants
#define MULTI 1

//////////// SETUP ////////////////////

#define Sx 21
#define Sy 21
#define S (Sx*Sy)   	// world size
#define A 4     	// number of actions
#define C 2        // if agent is within visible region
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

double euclidean(int agent, int goal){
	double dist;

	dist = sqrt(pow((agent%Sx - goal%Sx),2) + pow((agent/Sx - goal/Sx),2));
}

///////////// CODE ////////////////////////

int main() {
	long t=0L;

	double eta0=0.1,eta,gamma=0.9,eps, eps0=0.50;
  // Agent 1 values
  int s, s0_1,s1_1,a,a0_1,a1_1=4,ar_1,u;
	int hi_1;
	double rr_1;
  double ent_1, qq_1;
	double V_1[S],qn_1,qa_1[S][A], qx_1[A];
	double mu_1[S];
  int manhattan_dist_1;
	double euclidean_dist_1;
  double mmu_1;
  // Agent 2 values
#if MULTI
  int s0_2,s1_2,a0_2,a1_2=4,ar_2;
	int hi_2;
  double rr_2;
  double V_2[S],qn_2,qa_2[S][A], qx_2[A];
  double mu_2[S];
  int manhattan_dist_2;
	double euclidean_dist_2;
  double mmu_2;
#endif
	int features[S][A] = {0};
	int cor = 0;
  int collision;

	FILE *dat_1, *dat_2, *datcomb,*mud_1, *mud_2, *mudcomb;

	eta=eta0;
  eps=eps0;

//// Initialise value functions /////////////////
	for (s=0;s<S;s++) {		// init value function
		for (a=0;a<A;a++) {
      if (cheobs(s)==1) {
        qa_1[s][a]=0.0;
#if MULTI
        qa_2[s][a]=0.0;
#endif
      }
			else {
        qa_1[s][a]=drand48();
#if MULTI
        qa_2[s][a]=drand48();
#endif
		  }
		  if (cheobs(s)==1) {
      mu_1[s]=-1000.0;
#if MULTI
      mu_2[s]=-1000.0;
#endif
      }
		  else {
      mu_1[s]=1.0;
#if MULTI
      mu_2[s]=1.0;
#endif
    }
  }
}

/// Set sensor readings in features array ////////
///////////// Not setup for multi yet. Ignore //////////////////////////////
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
			s0_1=(int)(drand48()*(double)S);
		} while (cheobs(s0_1));

		a1_1=(int)(drand48()*(double)A);

#if MULTI
    do {
      s0_2=(int)(drand48()*(double)S);
		} while ((cheobs(s0_2)) && (s0_2 != s0_1));

    a1_2=(int)(drand48()*(double)A);
#endif
/////////// Start Episode ////////////////////////////////////////
		for (u=0;u<=2*Sx;u++) { // time steps per epoch for training
			//fprintf(dft,"%d %d\n",s0%Sx,s0/Sx);

			a0_1=a1_1;    //this action is now to be used
			ar_1=a0_1;   // agent 1

#if MULTI
      a0_2=a1_2;    //this action is now to be used
			ar_2=a0_2;   // agent 2

      collision = 0; // reset collision
#endif

			if (drand48()<eps) ar_1=(int)(drand48()*(double)A);
						// eps-greedy // agent 1
#if MULTI
      if (drand48()<eps) ar_2=(int)(drand48()*(double)A);
      			// eps-greedy // agent 2
#endif
			if ((s0_1<0)||(s0_1>=S)) {printf("check s0_1\n"); break;}
			s1_1=s0_1;
			rr_1=0.0;
			hi_1 = 0;
			// cor_1 = 0;

			switch(ar_1) {  // moving given action
				case 0: if (s0_1%Sx>0) s1_1=s0_1-1;
					break;
				case 1: if (s0_1/Sx<Sy-1) s1_1=s0_1+Sx;
					break;
				case 2:	if (s0_1%Sx<Sx-1) s1_1=s0_1+1;
					break;
				case 3:	if (s0_1/Sx>0) s1_1=s0_1-Sx;
					break;
				// case 4: hi_1 = 1;
			}

			mu_1[s0_1]+=1.0;

#if MULTI
			if (cheobs(s1_1)) s1_1=s0_2;

      if ((s0_2<0)||(s0_2>=S)) {printf("check s0_2\n"); break;}
			s1_2=s0_2;
			rr_2=0.0;
			hi_2 = 0;
			// cor_2 = 0;
			switch(ar_2) {  // moving given action
				case 0: if (s0_2%Sx>0) s1_2=s0_2-1;
					break;
				case 1: if (s0_2/Sx<Sy-1) s1_2=s0_2+Sx;
					break;
				case 2:	if (s0_2%Sx<Sx-1) s1_2=s0_2+1;
					break;
				case 3:	if (s0_2/Sx>0) s1_2=s0_2-Sx;
					break;
				// case 4: hi_2 = 1;
			}

			mu_2[s0_2]+=1.0;

			if (cheobs(s1_2)) s1_2=s0_2;

      // Collision Check, if attempt to occupy same square, returned to original
      if (s1_1 == s1_2) {
        s1_1 = s0_1;
        s1_2 = s0_2;
        collision = 1;
      }

#endif
///////// Reflexive Components Rewards ///////////////
//////// Entropy /////////////////////////////////////
//Entropy is is just for single agents
#if ENT
      // Agent 1 Entropy
      if (s1_1==s0_1) rr_1-=1.0;

      qq_1=0.0;
      for (a=0;a<A;a++) {
        qx_1[a]=exp(qa_1[s1_1][a]*5.1);
        qq_1+=qx_1[a];
      }
      ent_1=0.0;
      for (a=0;a<A;a++) {
        ent_1-=(qx_1[a]/qq_1)*log(qx_1[a]/qq_1);
      }
      rr_1+=ent_1;
#endif
// Commented out while doing pre-tests on entropy. Needs fixing for two agents
// ////// Gamma Entropy /////////////////////////////////
#if GAM
			if (s1_1==s0_1) rr_1=-1.0;
#endif
////// Wall favouring /////////////////////////////////
#if WAL
      for (a=0; a<A; a++){
        if (features[s1_1][a] ==1) cor+=1;
      }
      if (cor>0) rr_1 = 1.0;
#endif
////// Corner favouring //////////////////////////////
#if COR
			for (a=0; a<A; a++){
				if (features[s1_1][a] ==1) cor+=1;
			}
			if (cor>1) rr_1 = 1.0;
#endif

// Multi agent Rewards
#if MULTI
      manhattan_dist_1 = manhattan(s1_1, s1_2);
      manhattan_dist_2 = manhattan(s1_2, s1_1);
			//
      // if (manhattan_dist_1<5)	rr_1 = 1.0;
      // if (manhattan_dist_2<5)	rr_2 = 1.0;

			if ((hi_1)&&(hi_2)){
				if (manhattan_dist_1<3) {
					rr_1 = 1.0;
					rr_2 = 1.0;
				// }else{
				// 	rr_1 = -1.0;
				}
			}
			if (hi_1){
				if (manhattan_dist_1<3) {
					rr_1 = 1.0;
			// 	// } else {
			// 	// 	rr_2 = -1.0;
				}
			}

			if (hi_2){
				if (manhattan_dist_2<3) {
					rr_2 = 1.0;
			// 	// } else {
			// 	// 	rr_2 = -1.0;
				}
			}

			//
			if (collision) {
				rr_1 = 1.0;
				rr_2 = 1.0;
			}

			// euclidean_dist_1 = euclidean(s1_1, s1_2);
			// euclidean_dist_2 = euclidean(s1_2, s1_1);
			//
			// if (euclidean_dist_1>7) rr_1 = 1.0;
			// if (euclidean_dist_2>7) rr_2 = 1.0;
#endif

////// Action Choices /////////////////////////////////
    // Agent 1
			if ((s1_1<0)||(s1_1>=S)) {printf("check s1_1\n"); break;}

			qn_1=-1.0e10;
			for (a=0;a<A;a++) {
				if (qa_1[s1_1][a]>qn_1) {
					qn_1=qa_1[s1_1][a];
					a1_1=a;  // choose new action
				}
			}
			V_1[s1_1]=qn_1;   // value for "next" state

    // Agent 2
#if MULTI
      if ((s1_2<0)||(s1_2>=S)) {printf("check s1_2\n"); break;}


      qn_2=-1.0e10;
      for (a=0;a<A;a++) {
        if (qa_2[s1_2][a]>qn_2) {
          qn_2=qa_2[s1_2][a];
          a1_2=a;  // choose new action
        }
      }
      V_2[s1_2]=qn_2;   // value for "next" state
#endif

////// Update Value Function /////////////////////////////////
    // Agent 1
			qa_1[s0_1][a0_1]+=eta*(rr_1+gamma*V_1[s1_1]-qa_1[s0_1][a0_1]);
			s0_1=s1_1;

#if MULTI
    // Agent 2
      qa_2[s0_2][a0_2]+=eta*(rr_2+gamma*V_2[s1_2]-qa_2[s0_2][a0_2]);
      s0_2=s1_2;

#endif
		}

///// End Episode /////////////////////////////////////////////
		if (t%TM==TM-1) { // print out, but not in every step
			eps=eps0*((double)(T-t)/(double)T)*((double)(T-t)/(double)T);
			if (eps<0.01) eps=0.01;

			eta=eta0*((double)(T-t)/(double)T)*((double)(T-t)/(double)T);
			printf("%ld %g %g\n",t,eta,eps);

      // Agent 1
			if ((dat_1=fopen("dat_1","wt"))==NULL) printf("file error\n");
			if ((mud_1=fopen("mud_1","wt"))==NULL) printf("file error\n");
			for(s=0;s<S;s++) {
				qn_1=-1.0e10;
				for (a=0;a<A;a++) {
					if (qa_1[s][a]>qn_1) {
						qn_1=qa_1[s][a];
						a1_1=a;
					}
				}
				V_1[s]=qn_1;
				fprintf(dat_1,"%d %d %g\n",s%Sx,s/Sx,V_1[s]);
				if (s%Sx==Sx-1) fprintf(dat_1,"\n");
			}

      // Agent 1 density
      mmu_1=0.0;
      for(s=0;s<S;s++) {
        mmu_1+=mu_1[s];
      }
      mmu_1/=(double)S;
      for(s=0;s<S;s++) {
        fprintf(mud_1,"%d %d %g\n",s%Sx,s/Sx,mu_1[s]/mmu_1);
        if (s%Sx==Sx-1) fprintf(mud_1,"\n");
      }
      fclose(dat_1);
      fclose(mud_1);
#if MULTI
      // Agent 2
			if ((dat_2=fopen("dat_2","wt"))==NULL) printf("file error\n");
			if ((mud_2=fopen("mud_2","wt"))==NULL) printf("file error\n");
			for(s=0;s<S;s++) {
				qn_2=-1.0e10;
				for (a=0;a<A;a++) {
					if (qa_2[s][a]>qn_2) {
						qn_2=qa_2[s][a];
						a1_2=a;
					}
				}
				V_2[s]=qn_2;
				fprintf(dat_2,"%d %d %g\n",s%Sx,s/Sx,V_2[s]);
				if (s%Sx==Sx-1) fprintf(dat_2,"\n");
			}
      // fprintf(dat_2, "Its after here cunt\n");

      // Agent 1 density
      mmu_2=0.0;
      for(s=0;s<S;s++) {
        mmu_2+=mu_2[s];
      }
      mmu_2/=(double)S;
      for(s=0;s<S;s++) {
        fprintf(mud_2,"%d %d %g\n",s%Sx,s/Sx,mu_2[s]/mmu_2);
        if (s%Sx==Sx-1) fprintf(mud_2,"\n");
      }
      fclose(dat_2);
      fclose(mud_2);
#endif
		}
  }

	fflush(stdout);
	return(1);
}
