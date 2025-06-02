#include "detector.h"

TrafficDetector::TrafficDetector(ros::NodeHandle& nh) : nh_(nh) {
    // 初始化订阅者和发布者
    image_sub_ = nh_.subscribe("/camera/image_raw", 1, 
                              &TrafficDetector::imageCallback, this);
    cross_pub_ = nh_.advertise<std_msgs::Int16>("/detector/cross/result", 1);
    light_pub_ = nh_.advertise<std_msgs::Int16>("/detector/light/result", 1);
    image_pub_ = nh_.advertise<sensor_msgs::Image>("/detector_try/processed_image", 1);
    
    // 从参数服务器读取参数
    nh_.param("cross_threshold", cross_threshold, 0.7);
    nh_.param("circle_threshold", circle_threshold, 0.7);
    nh_.param("light_threshold", light_threshold, 200);
    nh_.param("min_contour_area", min_contour_area, 100);
}

void TrafficDetector::imageCallback(const sensor_msgs::ImageConstPtr& msg) {
    try {
        // 将ROS图像消息转换为OpenCV格式
        cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::Mat image = cv_ptr->image;
        
        // 检测斑马线
        bool cross_detected = detectCrosswalk(image);
        std_msgs::Int16 cross_msg;
        cross_msg.data = cross_detected ? 1 : 0;
        cross_pub_.publish(cross_msg);
        
        // 检测红绿灯
        int light_state = detectTrafficLight(image);
        std_msgs::Int16 light_msg;
        light_msg.data = light_state;
        light_pub_.publish(light_msg);
        
        // 可视化结果
        std::vector<std::vector<cv::Point>> red_contours, yellow_contours;
        detect_traffic_binary(red_binary, gray, red_contours);
        detect_traffic_binary(yellow_binary, gray, yellow_contours);
        visualizeResults(image, red_contours, yellow_contours, cross_detected);
        
    } catch (cv_bridge::Exception& e) {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

bool TrafficDetector::detectCrosswalk(const cv::Mat& image) {
    cv::Mat processed = preprocessImage(image);
    cv::Mat gray;
    cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);
    
    // 二值化
    cv::Mat binary;
    cv::threshold(gray, binary, 127, 255, cv::THRESH_BINARY);
    
    // 查找方形轮廓
    std::vector<cv::Rect> contours = findContours(binary);
    
    for (const auto& rect : contours) {
        // 检查矩形是否符合斑马线的特征
        double aspect_ratio = static_cast<double>(rect.width) / rect.height;
        if (aspect_ratio > 2.0 && rect.area() > min_contour_area) {
            return true;
        }
    }
    
    return false;
}

int TrafficDetector::detectTrafficLight(const cv::Mat& image) {
    cv::Mat processed = preprocessImage(image);
    cv::Mat gray;
    cv::cvtColor(processed, gray, cv::COLOR_BGR2GRAY);
    
    // 分离颜色通道
    std::vector<cv::Mat> channels;
    cv::split(processed, channels);
    
    cv::Mat red_channel = channels[2] - channels[1];
    cv::Mat yellow_channel = channels[1] - channels[0];
    
    // 二值化
    cv::Mat red_binary, yellow_binary;
    cv::threshold(red_channel, red_binary, 50, 255, cv::THRESH_BINARY);
    cv::threshold(yellow_channel, yellow_binary, 50, 255, cv::THRESH_BINARY);
    
    // 检测圆形轮廓
    std::vector<std::vector<cv::Point>> red_contours, yellow_contours;
    detect_traffic_binary(red_binary, gray, red_contours);
    detect_traffic_binary(yellow_binary, gray, yellow_contours);
    
    if (!red_contours.empty()) {
        return 2; // 红灯
    }
    if (!yellow_contours.empty()) {
        return 1; // 黄灯
    }
    return 0; // 绿灯
}

void TrafficDetector::detect_traffic_binary(const cv::Mat& binary, const cv::Mat& gray,
                                          std::vector<std::vector<cv::Point>>& contours_result) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    for (const auto& contour : contours) {
        double area = cv::contourArea(contour);
        if (area < min_contour_area) {
            continue;
        }

        // 圆形度检测
        double circularity = calculateCircularity(contour);
        if (circularity < circle_threshold) { 
            continue;
        }

        // 最小外接圆检测
        cv::Point2f center;
        float radius;
        cv::minEnclosingCircle(contour, center, radius);
        
        // 中心点亮度检测
        if (gray.at<uchar>(center) < light_threshold) {
            continue;
        }

        contours_result.push_back(contour);
    }
}

double TrafficDetector::calculateCircularity(const std::vector<cv::Point>& contour) {
    double area = cv::contourArea(contour);
    double perimeter = cv::arcLength(contour, true);
    return 4 * CV_PI * area / (perimeter * perimeter);
}

cv::Mat TrafficDetector::preprocessImage(const cv::Mat& image) {
    cv::Mat processed;
    cv::resize(image, processed, cv::Size(), 0.5, 0.5);
    cv::GaussianBlur(processed, processed, cv::Size(5, 5), 0);
    
    return processed;
}

std::vector<cv::Rect> TrafficDetector::findContours(const cv::Mat& binary) {
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    
    cv::findContours(binary, contours, hierarchy, 
                     cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    
    std::vector<cv::Rect> rects;
    for (const auto& contour : contours) {
        cv::Rect rect = cv::boundingRect(contour);
        if (rect.area() > min_contour_area) {
            rects.push_back(rect);
        }
    }
    
    return rects;
}

void TrafficDetector::visualizeResults(const cv::Mat& image, 
                                     const std::vector<std::vector<cv::Point>>& red_contours,
                                     const std::vector<std::vector<cv::Point>>& yellow_contours,
                                     bool cross_detected) {
    cv::Mat display = image.clone();
    
    // 绘制红绿灯检测结果
    for (const auto& contour : red_contours) {
        cv::drawContours(display, std::vector<std::vector<cv::Point>>{contour}, 0, 
                        cv::Scalar(0, 0, 255), 2);
        cv::putText(display, "Red Light", contour[0], 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
    }
    
    for (const auto& contour : yellow_contours) {
        cv::drawContours(display, std::vector<std::vector<cv::Point>>{contour}, 0, 
                        cv::Scalar(0, 255, 255), 2);
        cv::putText(display, "Yellow Light", contour[0], 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255), 2);
    }
    
    // 绘制斑马线检测结果
    if (cross_detected) {
        cv::putText(display, "Crosswalk Detected", cv::Point(10, 30), 
                   cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    }
    
    // 发布处理后的图像
    sensor_msgs::ImagePtr msg = 
        cv_bridge::CvImage(std_msgs::Header(), "bgr8", display).toImageMsg();
    image_pub_.publish(msg);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "traffic_detector");
    ros::NodeHandle nh;
    
    TrafficDetector detector(nh);
    
    ros::spin();
    return 0;
}
