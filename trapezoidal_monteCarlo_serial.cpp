//serial code
#include<iostream>
#include<math.h>
#include<cstdlib>
#include<ctime>
#include<time.h>
#include<random>
using namespace std;

//trapezoidal rule
double trap1(double lb, double ub, double n){
    double h=(ub-lb)/n;
    double prev=lb,now = lb+h;
    double ans=0.0;
    for(int i=1;i<=(int)n;i++){
        ans+=cos(now)+cos(prev); //sum of parallel sides
        prev=now;
        now=prev+h;
    }
    return 0.5*h*ans; //area = 0.5*distance between parallel sides*sum of parallel sides
}
//to use parallelism, need to make each value in loop independent
double trap2(double lb, double ub, double n){
    double h=(ub-lb)/n; //distance between two nearest partition points
    double ans=0.0; //final answer
    for(int i=1;i<=(int)n;i++){
        ans+=cos(lb+(i-1)*h)+cos(lb+i*h); //sum of parallel sides term added here
    }
    return 0.5*h*ans; //0.5*h is 0.5*distance between the parallel sides, in the formula for trapezoid area calculation
}


//monte carlo method
double monteC(double lb, double ub, double n){
    clock_t startm = clock();
    double x=0,y=0; //x,y coordinate of random number
    double xrange=ub-lb,yrange=1; //limits of x and y range (y varies from 0 to 1)
    int num=0,den=0; //numerator and denominator for final calculation
    int i=1; 
    static uniform_real_distribution<> distx(lb,nextafter(ub, numeric_limits<float>::max())); //x distribution, nextafter() is used because with distx(lb,ub) [lb,ub) numbers will be picked up. Added nextafter() takes [lb,ub] interval values
    static uniform_real_distribution<> disty(0,nextafter(1, numeric_limits<float>::max())); //y distribution
    static default_random_engine ans;
    #pragma omp parallel for private(i,x,y) reduction(+:num)
    for(i=1;i<=(int)n;i++){
        /*
        //following commented lines are for the rand() function which was used earlier, but failed
        x=lb+(((double)rand()/(double)RAND_MAX)*xrange); //scaling values within the range
        y=(double)rand()/(double)RAND_MAX; //(double)rand()/(double)RAND_MAX)*1
        stringstream stream; // since cout is not thread safe, this was used so that all values be printed in the same line. Else it gets printed whenever any thread reaches at that position, which gives a not-so-decent output.
        stream << x << " " << y << " " << omp_get_thread_num() <<endl;
        cout << stream.str();
        */
        x=distx(ans);
        y=disty(ans);
        if((cos(x)-y)>0) num++;
        //den++; this value will be equal to n, so don't need to waste computation power and space for it
    }
    //cout<<"monte carlo time "<<((double)(clock()-startm)/CLOCKS_PER_SEC)<<" sec"<<endl;
    return M_PI*(double)num/n;
    
}

int main(){
    clock_t start = clock();
    double lb=-1*M_PI*0.5,ub=M_PI*0.5; //lower bound and upper bound specified
    //50 = helping define the no of partition pts, 
    //accuracy of result obtained will increase with increased number of partition points
    //cout<<cos(lb)<<" "<<cos(ub)<<endl; //checking
    srand(time(0)); //seeding the random number generator
    //cout<<"Trapezoidal rule1 result = "<<trap1(lb,ub,50.0)<<endl;
    cout<<"Trapezoidal rule2 result = "<<trap2(lb,ub,50.0)<<endl;
    cout<<"Monte carlo method result = "<<monteC(lb,ub,50.0)<<endl;
    cout<<"Time for execution is "<<((double)(clock()-start)/CLOCKS_PER_SEC)<<" sec"<<endl;
    return 0;
}