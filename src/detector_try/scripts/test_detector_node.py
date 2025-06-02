#!/usr/bin/env python3

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Int16
from cv_bridge import CvBridge

class DetectorNodeTester:
    def __init__(self):
        # 初始化ROS节点
        rospy.init_node('detector_node_tester', anonymous=True)
        
        # 创建发布者和订阅者
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=1)
        self.cross_sub = rospy.Subscriber('/detector/cross/result', Int16, self.cross_callback)
        self.light_sub = rospy.Subscriber('/detector/light/result', Int16, self.light_callback)
        
        # 创建CV Bridge
        self.bridge = CvBridge()
        
        # 初始化测试结果
        self.test_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tests": []
        }
        
        # 创建测试结果目录
        if not os.path.exists("test_results"):
            os.makedirs("test_results")
            
        # 初始化检测结果
        self.cross_result = None
        self.light_result = None
        
    def cross_callback(self, msg):
        """斑马线检测结果回调"""
        self.cross_result = msg.data
        
    def light_callback(self, msg):
        """红绿灯检测结果回调"""
        self.light_result = msg.data
        
    def save_test_image(self, test_name, frame):
        """保存测试图像"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"test_results/{test_name}_{timestamp}.jpg"
        cv2.imwrite(filename, frame)
        return filename
        
    def run_test(self, test_case):
        """运行单个测试"""
        print(f"\n开始测试: {test_case['name']}")
        print(f"描述: {test_case['description']}")
        print(f"预期结果: {test_case['expected_results']}")
        
        # 读取视频文件
        video_path = os.path.join(os.path.dirname(__file__), "..", test_case["video_path"])
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"错误：无法打开视频文件 {video_path}")
            return
            
        # 获取视频帧率
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 计算时间范围对应的帧范围
        time_range = test_case.get("time_range", [0, 0])
        start_frame = int(time_range[0] * fps)
        end_frame = int(time_range[1] * fps) if time_range[1] > 0 else int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 设置起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        self.current_test = {
            "name": test_case["name"],
            "description": test_case["description"],
            "video_path": test_case["video_path"],
            "expected_results": test_case["expected_results"],
            "time_range": time_range,
            "frames": []
        }
        
        frame_count = start_frame
        while cap.isOpened() and frame_count < end_frame and not rospy.is_shutdown():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 发布图像到ROS话题
            try:
                ros_image = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.image_pub.publish(ros_image)
            except Exception as e:
                print(f"发布图像时出错: {e}")
                continue
                
            # 等待检测结果
            rate = rospy.Rate(10)  # 10Hz
            for _ in range(10):  # 等待最多1秒
                if self.cross_result is not None and self.light_result is not None:
                    break
                rate.sleep()
                
            # 可视化结果
            result_frame = self.visualize_results(frame)
            
            # 显示当前时间
            current_time = frame_count / fps
            cv2.putText(result_frame, f"Time: {current_time:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 保存结果
            frame_result = {
                "frame_number": frame_count,
                "time": current_time,
                "cross_detected": bool(self.cross_result),
                "light_state": self.light_result,
                "image_path": self.save_test_image(
                    f"test_{len(self.test_results['tests'])}", 
                    result_frame
                )
            }
            self.current_test["frames"].append(frame_result)
            frame_count += 1
            
            # 每100帧检查一次结果
            if len(self.current_test["frames"]) % 100 == 0:
                self.analyze_results()
                
            # 显示处理后的图像
            cv2.imshow('Test Result', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
            # 重置检测结果
            self.cross_result = None
            self.light_result = None
                
        cap.release()
        cv2.destroyAllWindows()
        
        # 保存测试结果
        self.test_results["tests"].append(self.current_test)
        self.save_test_results()
        
    def visualize_results(self, frame):
        """可视化检测结果"""
        display = frame.copy()
        
        # 绘制斑马线检测结果
        if self.cross_result:
            cv2.putText(display, "Crosswalk Detected", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # 绘制红绿灯检测结果
        if self.light_result is not None:
            text = "Red Light" if self.light_result == 2 else \
                  "Yellow Light" if self.light_result == 1 else \
                  "Green Light"
            color = (0, 0, 255) if self.light_result == 2 else \
                   (0, 255, 255) if self.light_result == 1 else \
                   (0, 255, 0)
            cv2.putText(display, text, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        
        return display
        
    def analyze_results(self):
        """分析当前测试结果"""
        if not self.current_test["frames"]:
            return
            
        # 计算检测准确率
        cross_correct = 0
        light_correct = 0
        total_frames = len(self.current_test["frames"])
        
        for frame in self.current_test["frames"]:
            if frame["cross_detected"] == self.current_test["expected_results"]["cross"]:
                cross_correct += 1
            if frame["light_state"] == self.current_test["expected_results"]["light"]:
                light_correct += 1
                
        cross_accuracy = cross_correct / total_frames
        light_accuracy = light_correct / total_frames
        
        print(f"\n当前测试统计:")
        print(f"斑马线检测准确率: {cross_accuracy:.2%}")
        print(f"红绿灯检测准确率: {light_accuracy:.2%}")
        
    def save_test_results(self):
        """保存所有测试结果到JSON文件"""
        filename = f"test_results/test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(self.test_results, f, indent=4)
        print(f"\n测试报告已保存到: {filename}")

def load_test_cases():
    """加载测试用例配置"""
    config_path = os.path.join(os.path.dirname(__file__), "..", "config", "test_cases.json")
    with open(config_path, 'r') as f:
        return json.load(f)["test_cases"]

def main():
    tester = DetectorNodeTester()
    
    # 加载测试用例
    test_cases = load_test_cases()
    
    try:
        for test_case in test_cases:
            tester.run_test(test_case)
            time.sleep(2)  # 测试间隔
            
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    finally:
        tester.save_test_results()
        
if __name__ == '__main__':
    main() 