#include<iostream>
#include<cstdlib>
#include<climits>
#include<mpi.h>
#include<time.h>
#include<random>
#include<sstream>
#include<vector>
using namespace std;

#define ROOT 0

int main(int argc, char *argv[]){
	int n=100;
	float a[n][n], b[n][n];
	int npes,mype,i=0,j=0,k=0,m=0;
	float start=0.0,temp=0.0;
	float** xx = new float*[2];

    for (int i = 0; i < 2; i++)
        xx[i] = new float[2];
	vector<int> proc_allot(n,0); //allotment vector for rows to process

	int provided;
	MPI_Status* status;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	if (provided < MPI_THREAD_MULTIPLE) {
	   // Error - MPI does not provide needed threading level
		cout<<"error"<<endl;
	}
	MPI_Comm_size(MPI_COMM_WORLD, &npes);
  	MPI_Comm_rank(MPI_COMM_WORLD, &mype);

  	double st=MPI_Wtime();  
  	vector<int> rows(npes,0);
  	int remaining=n%npes;//remaining extra rows to be distributed 
	//sending matrix a in cyclical manner for optimisation
  	for(i=0;i<n;i++){
  		proc_allot[i]=i%npes; //the process id that any row of matrix a is being allotted to
  	}
  	for(i=0;i<npes;i++){
		if(i<remaining) rows[i]=(n/npes)+1;
		else rows[i]=n/npes;
	}


  	if(mype==0){
  		//initialise matrices
  		static uniform_real_distribution<> dist(0,1); //distribution, normalised, can be multiplied according to range requirement 
	    static default_random_engine ans;
	    float x=0.0,y=0.0;
	    for(i=0;i<n;i++){
	    	for(j=0;j<n;j++){
	    		x=dist(ans);
	    		y=dist(ans);
	    		a[i][j]=x+0.1;
	    		b[i][j]=y+0.1;
	    	}
	    }
  	}
	// //sending matrices to all
  	MPI_Barrier(MPI_COMM_WORLD);
    MPI_Bcast(b,n*n,MPI_FLOAT,ROOT,MPI_COMM_WORLD); //sending matrix b
    MPI_Bcast(a,n*n,MPI_FLOAT,ROOT,MPI_COMM_WORLD); //sending matrix a
  	MPI_Barrier(MPI_COMM_WORLD);
	

  	//multiplying the matrix, compute c
  	//individual matrix get their own c
	float c_rec[rows[mype]][n];
  	for(i=0;i<rows[mype];i++){
    	for(j=0;j<n;j++){
			for(k=0;k<n;k++){
				c_rec[i][j]+=a[i*npes+mype][k]*b[k][j];
			}
		}
	}
	float** finalc = new float*[n]; //final c matrix
    	for (int i = 0; i < n; i++)
        finalc[i] = new float[n];
	float** c = new float*[n];
    for (int i = 0; i < n; i++)
        c[i] = new float[n];
	for(i=0;i<n;i++)
  		for(j=0;j<n;j++)
			c[i][j]=c_rec[i][j];


  	//printing final matrix c
  	//checking if the computed multiplication is correct
  	if(mype!=0){//send c
  		for(i=0;i<rows[mype];i++)
  			MPI_Send(c[i],n,MPI_FLOAT,0,i*npes+mype,MPI_COMM_WORLD);
  	}
  	if(mype==0){ //check
  		/*
  		vector<vector<float> > check(n,vector<float> (n,0));
		for(i=0;i<n;i++){
			for(j=0;j<n;j++){
				for(k=0;k<n;k++){
					check[i][j]+=a[i][k]*b[k][j];
				}
			}
		}
		*/
		for(i=0;i<n;i++){
			if(i%npes==0){
				for(j=0;j<n;j++) finalc[i][j]=c[i][j];
			}
			else MPI_Recv(finalc[i],n,MPI_FLOAT,proc_allot[i],i,MPI_COMM_WORLD,status);
		}
		//can check final matrix or their difference
		// for(int i=0;i<n;i++){
		// 	for(int j=0;j<n;j++){
		// 		cout<<finalc[i][j]<<" ";
		// 	}
		// 	cout<<endl;
		// }
		// cout<<endl;
  	}


  	
	float pivot[n];
	for(j=0;j<n;j++){ //which col has to be eliminated

		//make a copy of pivot row
		if(mype==proc_allot[j])
			for(i=0;i<n;i++) pivot[i]=c[(j-proc_allot[j])/npes][i];
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Bcast(pivot, n, MPI_FLOAT, proc_allot[j], MPI_COMM_WORLD); //broadcast pivot row for all to reduce
		MPI_Barrier(MPI_COMM_WORLD);

		//eliminate the row and change rest of the row
		for(i=0;i<rows[mype];i++){
			if(pivot[j]==0) continue; //could pivot
			if((i*npes+mype)>j){
				temp=c[i][j]/pivot[j];
				c[i][j]=0;
				for(k=j+1;k<n;k++){
					c[i][k]-=temp*pivot[k];
				}
			} 
		}
		MPI_Barrier(MPI_COMM_WORLD);
	}

	//send c to be compiled
  		for(i=0;i<rows[mype];i++)
  			MPI_Send(c[i],n,MPI_FLOAT,0,i*npes+mype,MPI_COMM_WORLD);
  	MPI_Barrier(MPI_COMM_WORLD);
  	//compile final c
  	if(mype==0){ //check
		for(i=0;i<n;i++){
			MPI_Recv(finalc[i],n,MPI_FLOAT,proc_allot[i],i,MPI_COMM_WORLD,status);
		}
		/* print
		for(int i=0;i<n;i++){
			for(int j=0;j<n;j++){
				cout<<finalc[i][j]<<" ";
			}
			cout<<endl;
		}
		cout<<endl;
		*/
		double end=MPI_Wtime();
		cout<<"Time taken by process "<<mype<<" is "<<(end - st)<<" sec"<<endl;
	}
	//double end=MPI_Wtime();
	//cout<<"Time taken by process "<<mype<<" is "<<(end - st)<<endl;
	MPI_Barrier(MPI_COMM_WORLD);//so that all finish work only then MPI finishes
  	MPI_Finalize(); 
    return 0;
}


	/*
	//code for partial pivoting, to be inserted inside for(j=0;j<n;j++), when iterating through columns
	//sending all rows below the diagonal, including the diagonal to master process
	for(i=0;i<rows[mype];i++){
		if((i*npes+mype)>=j)	MPI_Send(c[i],n,MPI_FLOAT,0,i*npes+mype,MPI_COMM_WORLD);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	//receiving all rows and finding the pivot value only by master process
	float maxval; //pivot value
	int maxrow,maxpe; //pivot row number in process, process number containing pivot row
	if(mype==0){
		for(i=j;i<n;i++) {
			MPI_Recv(finalc[i],n,MPI_FLOAT,proc_allot[i],i,MPI_COMM_WORLD,status);
			if(i==j){
				maxval=abs(finalc[i][j]);
				maxrow=(i-proc_allot[i])/npes;
				maxpe=proc_allot[i];
			}
			else {
				if(abs(finalc[i][j])>maxval){
					maxval=abs(finalc[i][j]);
					maxrow=(i-proc_allot[i])/npes;
					maxpe=proc_allot[i];
				}
			}
		}
	}

	//broadcasting all these values found
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Bcast(&maxval,1,MPI_FLOAT,ROOT,MPI_COMM_WORLD);
	MPI_Bcast(&maxrow,1,MPI_INT,ROOT,MPI_COMM_WORLD);
	MPI_Bcast(&maxpe,1,MPI_INT,ROOT,MPI_COMM_WORLD);

	if(maxval==0)	continue; //pivot value and all rows below it is zero

	//otherwise swap rows and proc_allot (since it will be used again for reducing further columns)
	if(mype==proc_allot[j]) MPI_Send(c[j],n,MPI_FLOAT,maxpe,100,MPI_COMM_WORLD);
	if(mype==maxpe){
		float temp[n];
		MPI_Recv(&temp[i],n,MPI_FLOAT,proc_allot[j],100,MPI_COMM_WORLD,status);
		MPI_Send(c[maxrow],n,MPI_FLOAT,proc_allot[j],200,MPI_COMM_WORLD);
		for(i=0;i<n;i++) c[maxrow][i]=temp[i];
	}
	if(mype==proc_allot[j]) MPI_Recv(c[j],n,MPI_FLOAT,maxpe,200,MPI_COMM_WORLD,status);
	MPI_Barrier(MPI_COMM_WORLD);
	proc_allot[maxrow*npes+maxpe]=proc_allot[j];
	proc_allot[j]=maxpe;
	*/