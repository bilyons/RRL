#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define OBS 0
#define TRI 0
#define TBC 1

#define ALT 0
#define SMV 0
#define EMP 0
#define WAL 0
#define COR 0
#define MID 1

#define Sx 21
#define Sy Sx
#define S (Sx*Sy)   	// world size
#define A 4        	// number of actions
#define T 10000000	// total epochs
#define TM 10000000

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

int main() {
	long t=0L;
	int s,s0,s1,a,a0,a1=4,ar,u;
	double eta0=0.1,eta,gamma=0.9,eps=0.75;
	double mmu;
	double rr;
	double V[S],qn,qa[S][A];
	double mu[S];
	int features[S][A] = {0};
	int cor = 0;
#if ALT
	long lt[S];
#endif
	FILE *dat,*mud;

	eta=eta0;
	srand48(time(0));

	for (s=0;s<S;s++) {		// init value function
		for (a=0;a<A;a++) {
			if (cheobs(s)==1) qa[s][a]=0.0;
			else qa[s][a]=drand48();
		}
		if (cheobs(s)==1) mu[s]=-1000.0;
		else mu[s]=1.0;


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

#if ALT
		lt[S]=-(long)(drand48()*(double)S);
#endif
	}

	for (t=0;t<T;t++) {		// loop over epochs
		do {
			s0=(int)(drand48()*(double)S);
		} while (cheobs(s0));

		a1=(int)(drand48()*(double)A);

		for (u=0;u<=2*Sx;u++) { // time steps per epoch (max is number of states)
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
#if ALT
			rr+=0.01*(double)(t-lt[s1]);
			lt[s1]=t;
#endif
			mu[s0]+=1.0;

			if (cheobs(s1)) s1=s0;

#if EMP
			if (s1==s0) rr=-1.0;
#endif

#if WAL
			if ((features[s1][0] == 1 ) || (features[s1][1] == 1 ) || (features[s1][2] == 1 ) || (features[s1][3] == 1 )) rr += 1.0;
#endif
#if COR
			for (a=0; a<A; a++){
				if (features[s1][a] ==1) cor+=1;
			}
			if (cor>=2) rr = 1.0;
#endif
#if MID
			for (a=0; a<A; a++){
				if (features[s1][a] ==1) cor+=1;
			}
			if (cor>1) rr = -5.0;
			if (cor ==1) rr = 1.0;

#endif
			if ((s1<0)||(s1>=S)) {printf("check s1\n"); break;}

			qn=-1.0e10;
			for (a=0;a<A;a++) {
				if (qa[s1][a]>qn) {
					qn=qa[s1][a];
					a1=a;  // choose new action
				}
			}
			V[s1]=qn;   // value for "next" state

#if SMV
			rr=1.0/(0.1+fabs(V[s1]-V[s0]));
#endif

			qa[s0][a0]+=eta*(rr+gamma*V[s1]-qa[s0][a0]);
			s0=s1;
		}


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
	fflush(stdout);
	return(1);
}
