#include <fstream>
#include <vector>
#include <iterator>
#include<chrono>
#include <opencv2/core.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include <iostream>
#include "Timer.h"
#include <iterator>
#include <future>
#include <set>

/*
 *
 * 1. Read vector and convert it into heatmaps (getOutputBlob) [Done]
 *                            |
 *                            Y  
 * 2. Resize heatmaps to appropriate dimensions (splitNetOutputBlobToParts) [Done]
 *                            |
 *                            Y
 * 3. Detect Keypoints based upon all the points which classify for threshold in resized heatmaps(getMultiPersonKeyPoints) [Done]
 *                            |
 *                            Y  
 * 4. Get valid and invalid pairs of keypoints for persons (getValidPairs) [Done]
 *                            |
 *                            Y
 * 5. Get keypoints for each person (getPersonWiseKeyPoints)  []
 */

const int nPoints = 18;
const int indexPoints_used[] = {1, 8, 9, 11, 12};
const int numPoints_used = sizeof(indexPoints_used) / sizeof(indexPoints_used[0]);

struct KeyPoint {
  KeyPoint(cv::Point point, float probability) {
    this->id = -1;
    this->point = point;
    this->probability = probability;
  }

  int id;
  cv::Point point;
  float probability;
};

struct ValidPair {
  ValidPair(int aId, int bId, float score) {
    this->aId = aId;
    this->bId = bId;
    this->score = score;
  }

  int aId;
  int bId;
  float score;
};

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

cv::Mat getOutputBlob(std::vector<float> &vector, int width, int height, int channels) {

  cv::Mat vec = cv::Mat(1, vector.size(), CV_32FC1, vector.data());
  std::vector<cv::Mat> channelVector(channels);

  for (int i = 0; i < channels; i++)
    channelVector[i] = vec.colRange(i * width * height, (i + 1) * width * height);

  for (cv::Mat &channel : channelVector)
    channel = channel.reshape(1, width);

  cv::Mat heatMap;
  cv::merge(channelVector, heatMap);

  return heatMap;

}

void getMultiPersonKeyPoints(cv::Mat &probMap, double threshold, std::vector<KeyPoint> &keyPoints) {

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

    keyPoints.push_back(KeyPoint(maxLoc, probMap.at<float>(maxLoc.y, maxLoc.x)));
  }

}

void getSinglePersonKeyPoint(cv::Mat &probMap, double threshold, cv::Point2i &keyPoint) {

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
//  for (auto i : indexPoints_used) {
    cv::Mat part = channels[i];

    cv::Mat resizedPart;

    cv::resize(part, resizedPart, targetSize);

    netOutputParts.push_back(resizedPart);
  }
}

void populateInterpPoints(const cv::Point &a, const cv::Point &b, int numPoints, std::vector<cv::Point> &interpCoords) {
  float xStep = ((float) (b.x - a.x)) / (float) (numPoints - 1);
  float yStep = ((float) (b.y - a.y)) / (float) (numPoints - 1);

  interpCoords.push_back(a);

  for (int i = 1; i < numPoints - 1; ++i) {
    interpCoords.push_back(cv::Point(a.x + xStep * i, a.y + yStep * i));
  }

  interpCoords.push_back(b);
}

void getValidPairs(const std::vector<cv::Mat> &netOutputParts,
                   const std::vector<std::vector<KeyPoint>> &detectedKeypoints,
                   std::vector<std::vector<ValidPair>> &validPairs,
                   std::set<int> &invalidPairs) {

  int nInterpSamples = 10;
  float pafScoreTh = 0.1;
  float confTh = 0.7;

  for (int k = 0; k < mapIdx.size(); ++k) {

    //A->B constitute a limb
    cv::Mat pafA = netOutputParts[mapIdx[k].first];
    cv::Mat pafB = netOutputParts[mapIdx[k].second];

    //Find the keypoints for the first and second limb
    const std::vector<KeyPoint> &candA = detectedKeypoints[posePairs[k].first];
    const std::vector<KeyPoint> &candB = detectedKeypoints[posePairs[k].second];

    int nA = candA.size();
    int nB = candB.size();

    /*
      # If keypoints for the joint-pair is detected
      # check every joint in candA with every joint in candB
      # Calculate the distance vector between the two joints
      # Find the PAF values at a set of interpolated points between the joints
      # Use the above formula to compute a score to mark the connection valid
     */

    if (nA != 0 && nB != 0) {
      std::vector<ValidPair> localValidPairs;

      for (int i = 0; i < nA; ++i) {
        int maxJ = -1;
        float maxScore = -1;
        bool found = false;

        for (int j = 0; j < nB; ++j) {
          std::pair<float, float> distance(candB[j].point.x - candA[i].point.x, candB[j].point.y - candA[i].point.y);

          float norm = std::sqrt(distance.first * distance.first + distance.second * distance.second);

          if (!norm) {
            continue;
          }

          distance.first /= norm;
          distance.second /= norm;

          //Find p(u)
          std::vector<cv::Point> interpCoords;
          populateInterpPoints(candA[i].point, candB[j].point, nInterpSamples, interpCoords);
          //Find L(p(u))
          std::vector<std::pair<float, float>> pafInterp;
          for (int l = 0; l < interpCoords.size(); ++l) {
            pafInterp.push_back(
                std::pair<float, float>(
                    pafA.at<float>(interpCoords[l].y, interpCoords[l].x),
                    pafB.at<float>(interpCoords[l].y, interpCoords[l].x)
                ));
          }

          std::vector<float> pafScores;
          float sumOfPafScores = 0;
          int numOverTh = 0;
          for (int l = 0; l < pafInterp.size(); ++l) {
            float score = pafInterp[l].first * distance.first + pafInterp[l].second * distance.second;
            sumOfPafScores += score;
            if (score > pafScoreTh) {
              ++numOverTh;
            }

            pafScores.push_back(score);
          }

          float avgPafScore = sumOfPafScores / ((float) pafInterp.size());

          if (((float) numOverTh) / ((float) nInterpSamples) > confTh) {
            if (avgPafScore > maxScore) {
              maxJ = j;
              maxScore = avgPafScore;
              found = true;
            }
          }

        }/* j */

        if (found) {
          localValidPairs.push_back(ValidPair(candA[i].id, candB[maxJ].id, maxScore));
        }
      }/* i */

      validPairs.push_back(localValidPairs);

    } else {
      invalidPairs.insert(k);
      validPairs.push_back(std::vector<ValidPair>());
    }
  }/* k */
}

void getPersonwiseKeypoints(const std::vector<std::vector<ValidPair>> &validPairs,
                            const std::set<int> &invalidPairs,
                            std::vector<std::vector<int>> &personwiseKeypoints) {
  for (int k = 0; k < mapIdx.size(); ++k) {
    if (invalidPairs.find(k) != invalidPairs.end()) {
      continue;
    }

    const std::vector<ValidPair> &localValidPairs(validPairs[k]);

    int indexA(posePairs[k].first);
    int indexB(posePairs[k].second);

    for (int i = 0; i < localValidPairs.size(); ++i) {
      bool found = false;
      int personIdx = -1;

      for (int j = 0; !found && j < personwiseKeypoints.size(); ++j) {
        if (indexA < personwiseKeypoints[j].size() &&
            personwiseKeypoints[j][indexA] == localValidPairs[i].aId) {
          personIdx = j;
          found = true;
        }
      }/* j */

      if (found) {
        personwiseKeypoints[personIdx].at(indexB) = localValidPairs[i].bId;
      } else if (k < 17) {
        std::vector<int> lpkp(std::vector<int>(18, -1));

        lpkp.at(indexA) = localValidPairs[i].aId;
        lpkp.at(indexB) = localValidPairs[i].bId;

        personwiseKeypoints.push_back(lpkp);
      }

    }/* i */
  }/* k */
}

void postprocessing(std::vector<float> &vector, int width, int height, int channels,
                    int frameWidth, int frameHeight, std::vector<std::vector<int> > &personwiseKeypoints) {

  //1. Get Heatmaps corresponding to each chanel


  Timer t1("1. Heatmap reshaping");
  cv::Mat heatMaps = getOutputBlob(vector, width, height, channels);
  t1.stop();

  Timer t2("2. Resizing heatmap");
  std::vector<cv::Mat> netOutputParts;
  splitNetOutputBlobToParts(heatMaps, cv::Size(frameWidth, frameHeight), netOutputParts);
  t2.stop();


  //3.Get the Keypoints
  Timer t3("3. Keypoints detection");
  int keyPointId = 0;
  std::vector<std::vector<KeyPoint>> detectedKeypoints;
  std::vector<KeyPoint> keyPointsList;

  for (int i = 0; i < nPoints; ++i) {
    std::vector<KeyPoint> keyPoints;

    getMultiPersonKeyPoints(netOutputParts[i], 0.1, keyPoints);

    for (int i = 0; i < keyPoints.size(); ++i, ++keyPointId) {
      keyPoints[i].id = keyPointId;
    }

    detectedKeypoints.push_back(keyPoints);
    keyPointsList.insert(keyPointsList.end(), keyPoints.begin(), keyPoints.end());
  }

  t3.stop();

  //4. Get valid invalid pairs
  Timer t4("4. Valid-invalid pairs search");
  std::vector<std::vector<ValidPair>> validPairs;
  std::set<int> invalidPairs;
  getValidPairs(netOutputParts, detectedKeypoints, validPairs, invalidPairs);
  t4.stop();

  //5. Get personwise keypoints
  Timer t5("4. Get person wise keypoints");
  getPersonwiseKeypoints(validPairs, invalidPairs, personwiseKeypoints);
  t5.stop();
}




