#!/usr/bin/env python3
import rospy
from std_msgs.msg import Int16
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

class DetectorTester:
    def __init__(self):
        rospy.init_node('detector_tester', anonymous=True)
        
        # 初始化订阅者
        self.cross_sub = rospy.Subscriber('/detector/cross/result', Int16, self.cross_callback)
        self.light_sub = rospy.Subscriber('/detector/light/result', Int16, self.light_callback)
        self.image_sub = rospy.Subscriber('/detector_try/processed_image', Image, self.image_callback)
        
        # 初始化变量
        self.cross_detected = False
        self.light_state = 0  # 0: 绿灯, 1: 黄灯, 2: 红灯
        self.bridge = CvBridge()
        
        # 初始化统计变量
        self.cross_count = 0
        self.light_count = {'green': 0, 'yellow': 0, 'red': 0}
        self.start_time = time.time()
        
    def cross_callback(self, msg):
        self.cross_detected = bool(msg.data)
        if self.cross_detected:
            self.cross_count += 1
            rospy.loginfo(f"斑马线检测到! 总计: {self.cross_count}")
    
    def light_callback(self, msg):
        self.light_state = msg.data
        if self.light_state == 0:
            self.light_count['green'] += 1
            rospy.loginfo("检测到绿灯")
        elif self.light_state == 1:
            self.light_count['yellow'] += 1
            rospy.loginfo("检测到黄灯")
        elif self.light_state == 2:
            self.light_count['red'] += 1
            rospy.loginfo("检测到红灯")
    
    def image_callback(self, msg):
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 显示图像
            cv2.imshow('Detection Result', cv_image)
            cv2.waitKey(1)
            
        except Exception as e:
            rospy.logerr(f"图像处理错误: {str(e)}")
    
    def print_statistics(self):
        elapsed_time = time.time() - self.start_time
        rospy.loginfo("\n=== 检测统计 ===")
        rospy.loginfo(f"运行时间: {elapsed_time:.2f} 秒")
        rospy.loginfo(f"斑马线检测次数: {self.cross_count}")
        rospy.loginfo("红绿灯检测统计:")
        rospy.loginfo(f"  绿灯: {self.light_count['green']}")
        rospy.loginfo(f"  黄灯: {self.light_count['yellow']}")
        rospy.loginfo(f"  红灯: {self.light_count['red']}")

def main():
    tester = DetectorTester()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        tester.print_statistics()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main() 