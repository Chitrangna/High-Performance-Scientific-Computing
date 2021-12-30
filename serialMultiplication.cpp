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
	vector<vector<float>> check(n,vector<float> (n,0));
	//initialise matrices
	static uniform_real_distribution<> dist(0,1); //distribution, normalised, can be multiplied according to range requirement 
    static default_random_engine ans;
    float x=0.0,y=0.0;
    int i=0,j=0,k=0;
    for(i=0;i<n;i++){
    	for(j=0;j<n;j++){
    		x=dist(ans);
    		y=dist(ans);
    		a[i][j]=x;
    		b[i][j]=y;
    	}
    }

	//matrix multiplication
	clock_t start = clock();
	float temp=0.0;
	for(i=0;i<n;i++){
		for(j=0;j<n;j++){
			temp=0.0;
			for(k=0;k<n;k++){
				temp+=a[i][k]*b[k][j];
			}
			check[i][j]=temp;
		}
	}
    cout<<"Time for multiplication is "<<(clock()-start)<<" sec"<<endl;
    
    return 0;
}	
