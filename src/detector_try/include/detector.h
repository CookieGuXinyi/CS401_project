#ifndef TRAFFIC_DETECTOR_H
#define TRAFFIC_DETECTOR_H

#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <std_msgs/Int16.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

class TrafficDetector {
public:
    TrafficDetector(ros::NodeHandle& nh);
    ~TrafficDetector() {}

private:
    // 回调函数
    void imageCallback(const sensor_msgs::ImageConstPtr& msg);
    
    // 检测函数
    bool detectCrosswalk(const cv::Mat& image);
    int detectTrafficLight(const cv::Mat& image);
    void detect_traffic_binary(const cv::Mat& binary, const cv::Mat& gray,
                             std::vector<std::vector<cv::Point>>& contours_result);
    
    // 图像处理辅助函数
    cv::Mat preprocessImage(const cv::Mat& image);
    std::vector<cv::Rect> findContours(const cv::Mat& binary);
    double calculateCircularity(const std::vector<cv::Point>& contour);
    void visualizeResults(const cv::Mat& image,
                         const std::vector<std::vector<cv::Point>>& red_contours,
                         const std::vector<std::vector<cv::Point>>& yellow_contours,
                         bool cross_detected);
    
    // ROS相关
    ros::NodeHandle nh_;
    ros::Subscriber image_sub_;
    ros::Publisher cross_pub_;
    ros::Publisher light_pub_;
    ros::Publisher image_pub_;
    
    // 参数
    double cross_threshold;
    double circle_threshold;
    double light_threshold;
    int min_contour_area;
};

#endif
