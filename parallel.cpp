#include<iostream>
#include <future>
#include <vector>
#include <algorithm>
#include <random>
#include "Timer.h"

typedef struct data{

    float max;
    int iD;

}groupData;


std::shared_ptr<groupData> find_max(std::vector<float> array,int iD){

    float max = array[0];
    int index = 0;
    for(int i = 1; i <array.size();i++){
        if (max < array[i]){
            max = array[i];
            // index = i;
        }
    
    }

    std::shared_ptr<groupData> d = std::make_shared<groupData>();
   
    d->iD = iD; d->max = max;
    return d;
}

float get_random() {
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0.1, 1); // rage 0 - 1
    return dis(e);
}

std::vector<float> get_random_vector(unsigned int size){

    std::vector<float> nums;

    for (int i{}; i != size; ++i) // Generate 5 random floats
        nums.push_back(get_random());

    return nums;
}




int main(){

    int size = 1000, N = 1000;

    std::vector< std::vector<float> > groupArrays;

    Timer t1("creating data");
    for(int i = 0; i < N; i++){
        std::vector<float> a = get_random_vector(size);
        groupArrays.push_back(a);
    }
    t1.stop();


    Timer t2("Non parallel sorting");
    for(int i = 0; i < N; i++){
        std::shared_ptr<groupData> res;
        res = find_max(groupArrays[i],i);
        // std::cout<<res->iD<<": "<<res->max<<std::endl;
    }
    t2.stop();

    


    return 0;
}
