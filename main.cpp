//
// Created by aman on 2019-11-11.
//


#include "utility.h"


#define NUM_CHANNELS 57
#define WIDTH 46
#define HEIGHT 46



int main(){

  std::vector<float> uarr = readData("/home/aman/frame_199.txt");
  cv::Mat input = cv::imread("/home/aman/frame_199.jpg", cv::IMREAD_COLOR);

  std::vector<int> results;
  postprocessing(uarr,WIDTH,HEIGHT,NUM_CHANNELS,input.cols,input.rows,results);

  for(auto &i : results)
    std::cout<<i<<std::endl;

  return 0;
}