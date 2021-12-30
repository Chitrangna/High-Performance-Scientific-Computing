//parallel code
#include<iostream>
#include<math.h>
#include<cstdlib>
#include<ctime>
#include<omp.h>
#include<time.h>
#include<random>
#include<sstream>
using namespace std;


//trapezoidal rule
//to use parallelism, need to make each value in loop independent
double trap2(double lb, double ub, double n){
    clock_t starttrap = clock();
    double h=(ub-lb)/n; //distance between two nearest partition points
    double ans=0.0; //final answer
    int i=1;
    double temp=0;
    #pragma omp parallel for private(i,temp) reduction(+:ans) //for parallelisation
    for(i=1;i<=(int)n;i++){
        double temp=cos(lb+(i-1)*h)+cos(lb+i*h); //so that threads wont have to wait for ans to be computed to get added to it
        ans+=temp; //sum of parallel sides term added here
        //ans+=cos(lb+(i-1)*h)+cos(lb+i*h);
    }
    cout<<"trapezoidal time  "<<((double)(clock()-starttrap)/CLOCKS_PER_SEC)<<" sec"<<endl;
    return 0.5*h*ans; //0.5*h is 0.5*distance between the parallel sides, in the formula for trapezoid area calculation
}

/*
void TrapezoidalParallel(double lb, double ub, double n){
    clock_t start = clock();
    int threads = 8; // number of threads implemented
    double width = abs((ub - lb)) / (double)threads; //width for each thread. This will be further broken down into trapezoids
    double result = 0; //final ans
    double h=(ub-lb)/n; //distance between top and base of various trapezoids to be made
    #pragma omp parallel for reduction(+:result)
    for(int i = 0; i < threads; i++){
        double x_from_val = i * width; //for this thread, start value of x
        double x_to_val = (i + 1) * width; //for this thread end value of x
        double sum = 0;        
        while(x_from_val < x_to_val) //individual breaking into trapezoid further
        {
            sum+=cos(lb+x_from_val)+cos(lb+x_from_val+h);
            x_from_val += h;
        }
        // #pragma omp critical
            result += sum;
    }
    cout<<"TrapezoidalParallel result: "<<0.5*h*result<<" with execution time: "<<((double)(clock()-start)/CLOCKS_PER_SEC)<<" sec"<<endl;
    return;
}
*/

//monte carlo method
double monteC(double lb, double ub, double n){
    clock_t startm = clock();
    double x=0,y=0; //x,y coordinate of random number
    double xrange=ub-lb,yrange=1; //limits of x and y range (y varies from 0 to 1)
    int num=0,den=0; //numerator and denominator for final calculation
    int i=1; 
    static uniform_real_distribution<> distx(lb,nextafter(ub, numeric_limits<float>::max())); //x distribution
    static uniform_real_distribution<> disty(0,nextafter(1, numeric_limits<float>::max())); //y distribution
    static default_random_engine ans;
    #pragma omp parallel for private(i,x,y) reduction(+:num)
    for(i=1;i<=(int)n;i++){
        x=distx(ans);
        y=disty(ans);
        if((cos(x)-y)>0) num++;
    }
    cout<<"monte carlo time "<<((double)(clock()-startm)/CLOCKS_PER_SEC)<<" sec"<<endl;
    return M_PI*(double)num/n;
    
}


int main(){
    clock_t start = clock();
    double lb=-1*M_PI*0.5,ub=M_PI*0.5; //lower bound and upper bound specified
    //50 = helping define the no of partition pts, 
    //accuracy of result obtained will increase with increased number of partition points
    //srand(time(0)); //seeding the random number generator
    double ans1 = trap2(lb,ub,50.0);
    double ans2 = monteC(lb,ub,50.0);
    cout<<"Trapezoidal rule2 result = "<<ans1<<endl;
    cout<<"Monte carlo method result = "<<ans2<<endl;
    cout<<"Time for execution is "<<((double)(clock()-start)/CLOCKS_PER_SEC)<<" sec"<<endl;
    return 0;
}