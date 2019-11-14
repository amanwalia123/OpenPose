//
// Created by aman on 2019-11-13.
//

#ifndef OPENPOSEKEYPOINTS_TIMER_H
#define OPENPOSEKEYPOINTS_TIMER_H

#include <chrono>
#include <iostream>
#include <string>

class Timer {
 public:
  Timer(const std::string &_message){

     m_StartTimePoint = std::chrono::high_resolution_clock::now();
     this->message = _message;

  }


  void stop(){
    auto endTimePoint = std::chrono::high_resolution_clock::now();
    auto start = std::chrono::time_point_cast<std::chrono::microseconds>(m_StartTimePoint).time_since_epoch().count();
    auto stop = std::chrono::time_point_cast<std::chrono::microseconds>(endTimePoint).time_since_epoch().count();

    auto  duration = stop - start;
    double ms = duration * 0.001;

    std::cout<<message<<":"<<duration<<"us ("<<ms<<"ms)"<<std::endl;
  }

 private:
  std::chrono::time_point<std::chrono::high_resolution_clock>  m_StartTimePoint;
  std::string message;
};

#endif //OPENPOSEKEYPOINTS_TIMER_H
