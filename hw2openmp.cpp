#include<iostream>
#include<math.h>
#include<cstdlib>
#include<climits>
#include<ctime>
#include<omp.h>
#include<time.h>
#include<random>
#include<sstream>
#include<vector>
using namespace std;

int main(){
	int n;
	cin>>n;//dimension of matrix
	vector<vector<float>> a(n,vector<float> (n,0));
	vector<vector<float>> b(n,vector<float> (n,0));
	vector<vector<float>> c(n,vector<float> (n,0));
	vector<vector<float>> check(n,vector<float> (n,0));
	//initialise matrices
	static uniform_real_distribution<> dist(0,1); //distribution, normalised, can be multiplied according to range requirement 
    static default_random_engine ans;
    float x=0.0,y=0.0;
    int i=0,j=0,k=0;
    #pragma omp parallel for private(i,j,x,y) shared(n,a,b)
    for(i=0;i<n;i++){
    	for(j=0;j<n;j++){
    		x=dist(ans);
    		y=dist(ans);
    		a[i][j]=x;
    		b[i][j]=y;
    	}
    }

	//matrix multiplication
	float temp=0.0;
	double start=omp_get_wtime();
	#pragma omp parallel for shared(n,a,b,c,check) private(i,j,k,temp) //reduction(+:temp)
	for(i=0;i<n;i++){
    	//#pragma omp parallel for private(j,k,temp) shared(n,a,b,c,check)
		for(j=0;j<n;j++){
			temp=0.0;
			for(k=0;k<n;k++){
				temp+=a[i][k]*b[k][j];
			}
			check[i][j]=temp;//this is c
			//to store as transpose to be used later, we do the following
			c[j][i]=temp; //transpose, for faster calculation later
		}
	}
    cout<<"Time for multiplication is "<<(omp_get_wtime()-start)<<" sec"<<endl;
    

//gaussian elimination
	start=omp_get_wtime();
	float r=0.0;
	int jmax=0;
	//make this upper triangular, method 1. 
	for(j=0;j<n;j++){
		//partial pivoting, finding max value below the diagonal 
		jmax=j;//diagonal element
		for(k=j+1;k<n;k++){
			if(abs(check[k][j])>abs(check[jmax][j])) jmax=k;
		}
		if(check[jmax][j]==0) continue; //all values below the diagonal are already zero
		else{
			//swap rows to bring pivot to diagonal element
			#pragma omp parallel for shared(n,check,jmax) private(temp,k)
			for(k=j;k<n;k++){
				temp=check[jmax][k];
				check[jmax][k]=check[j][k];
				check[j][k]=temp;
			}
			//make upper triangular
			#pragma omp parallel for shared(j,n,check) private(i,r,k)
			for(i=j+1;i<n;i++){
				r=check[i][j]/check[j][j];
				check[i][j]=0;
				for(k=j+1;k<n;k++){
					check[i][k]-=r*check[j][k];
				}
			}
		}
		
	}
	cout<<"Time for upper-triangulation with first method is "<<(omp_get_wtime()-start)<<" sec"<<endl;
	   

	//make it lower triangular - method 2
	start=omp_get_wtime();
	for(i=0;i<n;i++){
		//partial pivoting, finding max absolute value to the right of diagonal
		jmax=i;//diagonal element
		for(k=i+1;k<n;k++){
			if(abs(c[i][k])>abs(c[i][jmax])) jmax=k;
		}
		if(c[i][jmax]==0) continue;//upper triangle of the row is already zero
		else{
			//swap columns to bring pivot to diagonal element
			#pragma omp parallel for shared(n,c,jmax,i) private(temp,k)
			for(k=i;k<n;k++){
				temp=c[k][i];
				c[k][i]=c[k][jmax];
				c[k][jmax]=temp;
			}
			//making upper triangle zero and changing their column value accordingly
			#pragma omp parallel for shared(i,n,c) private(j,r,k)
			for(j=i+1;j<n;j++){
				r=c[i][j]/c[i][i];
				c[i][j]=0;
				for(k=i+1;k<n;k++){
					c[k][j]-=r*c[k][i];
				}
			}
		}
	}
	cout<<"Time for upper-triangulation with second method is "<<(omp_get_wtime()-start)<<" sec"<<endl;

	//transposing again to get upper triangular matrix
	#pragma omp parallel for shared(n,c) private(i,j)
	for(i=0;i<n;i++){
		for(j=i+1;j<n;j++){
			c[i][j]=c[j][i];
			c[j][i]=0;
		}
	}
	cout<<"Time for upper-triangulation/transpose with second method is "<<(omp_get_wtime()-start)<<" sec"<<endl;
    
    return 0;
}

//extra codes for checking
/*
	//checking if the computed multiplication is correct
	//vector<vector<float>> check(n,vector<float> (n,0));
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			for(k=0;k<n;k++){
				//double temp=;
				check[i][j]+=a[i][k]*b[k][j];
			}
		}
	}
	for(int i=0;i<n;i++){
		for(int j=0;j<n;j++){
			cout<<check[i][j]-c[j][i]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
	*/

/*
//matrix to see the difference between upper triangulated matrix by different methods
	cout<<"Matrix of difference between upper triangulated matrices with different methods "<<endl;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			cout<<c[i][j]-check[i][j]<<" ";
		}
		cout<<endl;
	}
	cout<<endl;
*/

