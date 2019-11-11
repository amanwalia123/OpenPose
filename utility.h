#include <fstream>
#include <vector>
#include <iterator>
#include<chrono>
#include <opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include <iostream>


const int nPoints = 18;

const std::string keypointsMapping[] = {
    "Nose", "Neck",
    "R-Sho", "R-Elb", "R-Wr",
    "L-Sho", "L-Elb", "L-Wr",
    "R-Hip", "R-Knee", "R-Ank",
    "L-Hip", "L-Knee", "L-Ank",
    "R-Eye", "L-Eye", "R-Ear", "L-Ear"
};

const std::vector<std::pair<int, int>> mapIdx = {
    {31, 32}, {39, 40}, {33, 34}, {35, 36}, {41, 42}, {43, 44},
    {19, 20}, {21, 22}, {23, 24}, {25, 26}, {27, 28}, {29, 30},
    {47, 48}, {49, 50}, {53, 54}, {51, 52}, {55, 56}, {37, 38},
    {45, 46}
};

const std::vector<std::pair<int, int>> posePairs = {
    {1, 2}, {1, 5}, {2, 3}, {3, 4}, {5, 6}, {6, 7},
    {1, 8}, {8, 9}, {9, 10}, {1, 11}, {11, 12}, {12, 13},
    {1, 0}, {0, 14}, {14, 16}, {0, 15}, {15, 17}, {2, 17},
    {5, 16}
};

std::vector<float> readData(const std::string &filename) {
  std::ifstream inputFile(filename);
  if (!inputFile.is_open()) {
    exit(-1);
  }
  std::istream_iterator<float> start(inputFile), end;
  return std::vector<float>(start, end);
}

cv::Mat getOutputBlob(std::vector<float> &vector, int width, int height, int channels) {

//  std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
  cv::Mat vec = cv::Mat(1, vector.size(), CV_32FC1, vector.data());
  std::vector<cv::Mat> channelVector(channels);

  for (int i = 0; i < channels; i++)
    channelVector[i] = vec.colRange(i * width * height, (i + 1) * width * height);

  for (cv::Mat &channel : channelVector)
    channel = channel.reshape(1, width);

  cv::Mat heatMap;
  cv::merge(channelVector, heatMap);
  std::chrono::time_point<std::chrono::system_clock> finishTP = std::chrono::system_clock::now();

//  std::cout << "Time Taken in reshaping vector = "
//            << std::chrono::duration_cast<std::chrono::milliseconds>(finishTP - startTP).count() << " ms" << std::endl;

  return heatMap;

}

void getKeyPoints(cv::Mat &probMap, double threshold, std::vector<cv::Point2d> &keyPoints) {

//  std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
  cv::Mat smoothProbMap;
  cv::GaussianBlur(probMap, smoothProbMap, cv::Size(3, 3), 0, 0);

  cv::Mat maskedProbMap;
  cv::threshold(smoothProbMap, maskedProbMap, threshold, 255, cv::THRESH_BINARY);

  maskedProbMap.convertTo(maskedProbMap, CV_8U, 1);

  std::vector<std::vector<cv::Point> > contours;
  cv::findContours(maskedProbMap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  for (int i = 0; i < contours.size(); ++i) {
    cv::Mat blobMask = cv::Mat::zeros(smoothProbMap.rows, smoothProbMap.cols, smoothProbMap.type());

    cv::fillConvexPoly(blobMask, contours[i], cv::Scalar(1));

    double maxVal;
    cv::Point maxLoc;

    cv::minMaxLoc(smoothProbMap.mul(blobMask), 0, &maxVal, 0, &maxLoc);

    keyPoints.push_back(maxLoc);
  }

//  std::chrono::time_point<std::chrono::system_clock> finishTP = std::chrono::system_clock::now();
//  std::cout << "Time Taken in getting keypoints = "
//            << std::chrono::duration_cast<std::chrono::milliseconds>(finishTP - startTP).count() << " ms" << std::endl;
}

void getSingleKeyPoint(cv::Mat &probMap, double threshold, cv::Point2i &keyPoint) {

//  std::chrono::time_point<std::chrono::system_clock> startTP = std::chrono::system_clock::now();
  cv::Mat smoothProbMap;
  cv::GaussianBlur(probMap, smoothProbMap, cv::Size(3, 3), 0, 0);

  cv::Mat maskedProbMap;
  cv::threshold(smoothProbMap, maskedProbMap, threshold, 255, cv::THRESH_BINARY);

  maskedProbMap.convertTo(maskedProbMap, CV_8U, 1);

  std::vector<std::vector<cv::Point> > contours;
  cv::findContours(maskedProbMap, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);

  double maxProb = 0.0;
  cv::Point2d loc;

  for (int i = 0; i < contours.size(); ++i) {
    cv::Mat blobMask = cv::Mat::zeros(smoothProbMap.rows, smoothProbMap.cols, smoothProbMap.type());

    cv::fillConvexPoly(blobMask, contours[i], cv::Scalar(1));

    double maxVal;
    cv::Point maxLoc;

    cv::minMaxLoc(smoothProbMap.mul(blobMask), 0, &maxVal, 0, &maxLoc);

    if (maxVal > maxProb) {
      loc = maxLoc;
      maxProb = maxVal;
    }

  }

  keyPoint = loc;

}

void splitNetOutputBlobToParts(cv::Mat &netOutputBlob,
                               const cv::Size &targetSize,
                               std::vector<cv::Mat> &netOutputParts) {
  int nParts = netOutputBlob.channels();
  int h = netOutputBlob.rows;
  int w = netOutputBlob.cols;

  cv::Mat channels[nParts];
  cv::split(netOutputBlob, channels);

  for (int i = 0; i < nParts; ++i) {
    cv::Mat part = channels[i];

    cv::Mat resizedPart;

    cv::resize(part, resizedPart, targetSize);

    netOutputParts.push_back(resizedPart);
  }
}

void postprocessing(std::vector<float> &vector, int width, int height, int channels,
                                        int frameWidth, int frameHeight,std::vector<int> &result) {

  //1. Get Heatmaps corresponding to each chanel
  cv::Mat heatMap = getOutputBlob(vector, width, height, channels);

  //2. Resize Heatmaps to dimension of image
  std::vector<cv::Mat> netOutputParts;
  splitNetOutputBlobToParts(heatMap, cv::Size(frameWidth, frameHeight), netOutputParts);

  //3.Get the Keypoints

  std::vector<cv::Point2i> detectedKeypoints;

  for (int i = 0; i < nPoints; ++i) {
    cv::Point2i keyPoint;
    getSingleKeyPoint(netOutputParts[i], 0.1, keyPoint);

      result.push_back(keyPoint.x);
      result.push_back(keyPoint.y);

  }
}
