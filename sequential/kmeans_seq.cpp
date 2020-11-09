#include "opencv2\opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

using namespace std;

class Point {
public:
    Point(int id, float b, float g, float r) {
        dimensions = 0;
        pointId = id;
        clusterId = INT32_MAX;
        values.push_back(b);
        values.push_back(g);
        values.push_back(r);
        dimensions = 3;
    }

    int getDimensions() {
        return dimensions;
    }

    int getCluster() {
        return clusterId;
    }

    int getId() {
        return pointId;
    }

    void setCluster(int id) {
        clusterId = id;
    }

    float getVal(int pos) {
        return values[pos];
    }

private:
    int pointId;
    int clusterId;
    int dimensions;
    vector<float> values;
};

class Cluster {
public:
    Cluster(int id, Point& centerPt) {
        clusterId = id;
        for (int i = 0; i < centerPt.getDimensions(); i++) {
            centroid.push_back(centerPt.getVal(i));
        }
        addPoint(centerPt);
    }

    void addPoint(Point& pt) {
        pt.setCluster(this->clusterId);
        points.push_back(pt);
    }

    bool removePoint(int pointId) {
        for (int i = 0; i < points.size(); i++) {
            if (points[i].getId() == pointId) {
                points.erase(points.begin() + i);
                return true;
            }
        }
        return false;
    }

    int getClusterId() {
        return clusterId;
    }

    Point getPoint(int pos) {
        return points[pos];
    }

    int getSize() {
        return points.size();
    }

    float getCentroidByPos(int pos) {
        return centroid[pos];
    }

    void setCentroidByPos(int pos, double val) {
        centroid[pos] = val;
    }

private:
    int clusterId;
    vector<float> centroid;
    vector<Point> points;
};

int getNearestCluster(vector<Cluster>& clusters, Point point, int k) {
    float dist;
    float minDist = FLT_MAX;
    int nearestClusterId = INT_MAX;
    int dimension = point.getDimensions();

    // Get distance from point to nearest centroid
    for (int i = 0; i < k; i++) {
        dist = 0.0;

        for (int j = 0; j < dimension; j++)
        {
            dist += pow(clusters[i].getCentroidByPos(j) - point.getVal(j), 2.0);
        }

        if (dist < minDist)
        {
            minDist = dist;
            nearestClusterId = clusters[i].getClusterId();
        }
    }
    return nearestClusterId;
}

cv::Mat kmeans_seq(cv::Mat image, int k, int iters) {
    cout << "---Sequential K-means---" << endl;

    int vecSize = image.rows * image.cols;
    vector<cv::Mat> chans;
    split(image, chans);
    cv::Mat res;
    for (int i = 0; i < chans.size(); i++) {
        chans[i] = chans[i].reshape(1, 1);
    }

    vector<unsigned char> uchar_pointVec;
    uchar_pointVec.assign(chans[0].data, chans[0].data + chans[0].total() * chans[0].channels());
    vector<float> chanVec0(uchar_pointVec.begin(), uchar_pointVec.end());
    uchar_pointVec.assign(chans[1].data, chans[1].data + chans[1].total() * chans[1].channels());
    vector<float> chanVec1(uchar_pointVec.begin(), uchar_pointVec.end());
    uchar_pointVec.assign(chans[2].data, chans[2].data + chans[2].total() * chans[2].channels());
    vector<float> chanVec2(uchar_pointVec.begin(), uchar_pointVec.end());

    vector<Point> pointVec;
    // Create point for each line
   for(int i = 0; i < vecSize; i++) {
        Point point(i, chanVec0[i], chanVec1[i], chanVec2[i]);
        pointVec.push_back(point);
    }

    int pointVecSize = pointVec.size();
    int dimension = pointVec[0].getDimensions();
    // Store points that are used in initializing clusters. Makes sure no repeat.
    vector<int> usedPoints;
    vector<Cluster> clusters;
    srand(time(0));
    for (int i = 0; i < k; i++) {
        while (true) {
            int idx = rand() % pointVecSize;

            if (find(usedPoints.begin(), usedPoints.end(), idx) == usedPoints.end()) {
                usedPoints.push_back(idx);
                pointVec[idx].setCluster(i);
                Cluster cluster(i, pointVec[idx]);
                clusters.push_back(cluster);
                break;
            }
        }
    }
    cout << clusters.size() << " clusters initialized" << endl;

    cout << "Running K-means clustering" << endl;
    auto start = chrono::high_resolution_clock::now();

    int iter = 0;

    while (iter < iters) {
        // check is true if point alr at best cluster
        bool check = true;

        // For each point, find nearest cluster and update cluster with point if move to new cluster
        for (int i = 0; i < pointVecSize; i++) {
            int curClusterId = pointVec[i].getCluster();
            int nearestClusterId = getNearestCluster(clusters, pointVec[i], k);

            if (curClusterId != nearestClusterId) {
                if (curClusterId != INT32_MAX) {
                    clusters[curClusterId].removePoint(pointVec[i].getId());
                }
                clusters[nearestClusterId].addPoint(pointVec[i]);
                check = false;
            }
        }

        // For each cluster, get new centroid
        for (int i = 0; i < k; i++) {
            int clusterSize = clusters[i].getSize();

            for (int j = 0; j < dimension; j++) {
                float sum = 0.0;
                if (clusterSize > 0) {
                    for (int p = 0; p < clusterSize; p++)
                        sum += clusters[i].getPoint(p).getVal(j);
                    clusters[i].setCentroidByPos(j, sum / clusterSize);
                }
            }
        }

        // Break if no change in all clusters
        if (check) {
            break;
        }
        else {
            iter++;
        }
    }

    auto end = chrono::high_resolution_clock::now();
    cout << "Clustering completed in iteration : " << iter << endl << endl;
    cout << "Time taken: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " microseconds" << endl;

    //for (int i = 0; i < k; i++) {
    //    cout << "Centroid in cluster " << clusters[i].getClusterId() << " : ";
    //    for (int j = 0; j < dimension; j++) {
    //        cout << clusters[i].getCentroidByPos(j) << " ";
    //    }
    //    cout << endl;
    //}

    int colSize = image.cols;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int curCluster = pointVec[(long int) i * colSize + j].getCluster();
            for (int dim = 0; dim < dimension; dim++) {
                image.at<cv::Vec3b>(i, j)[dim] = (unsigned char)clusters[curCluster].getCentroidByPos(dim);
            }
        }
    }
    return image;
}