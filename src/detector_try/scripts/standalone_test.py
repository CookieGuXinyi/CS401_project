#!/usr/bin/env python3

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime

class TrafficDetector:
    def __init__(self):
        # 检测参数
        self.cross_threshold = 0.7
        self.circle_threshold = 0.7
        self.light_threshold = 150  # 降低亮度阈值
        self.min_contour_area = 100
        self.stripe_threshold = 0.6  # 条纹检测阈值
        
    def preprocess_image(self, image):
        """图像预处理"""
        processed = cv2.resize(image, None, fx=0.5, fy=0.5)
        processed = cv2.GaussianBlur(processed, (5, 5), 0)
        return processed
        
    def detect_crosswalk(self, image):
        """检测斑马线"""
        processed = self.preprocess_image(image)
        
        # 1. 调整压缩比例
        processed = cv2.resize(processed, None, fx=0.5, fy=0.5)  # 提高分辨率
        
        # 2. 使用自适应阈值
        gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # 3. 使用更小的形态学核
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # 4. 改进轮廓筛选条件
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:  # 降低面积阈值
                continue
                
            # 使用最小外接矩形
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype=np.int32)
            
            # 检查位置（在图像下半部分）
            if rect[0][1] < 0.4 * processed.shape[0]:  # 放宽位置限制
                continue
                
            # 调整宽高比
            width = max(rect[1][0], rect[1][1])
            height = min(rect[1][0], rect[1][1])
            ratio = width / height
            
            if ratio < 3.0:  # 降低宽高比要求
                continue
                
            # 检查角度
            if abs(rect[2]) > 20:  # 放宽角度限制
                continue
                
            # 分析条纹特征
            roi = cv2.boundingRect(contour)
            x, y, w, h = roi
            if x < 0 or y < 0 or x + w > binary.shape[1] or y + h > binary.shape[0]:
                continue
                
            roi_binary = binary[y:y+h, x:x+w]
            stripes = self.analyze_stripes(roi_binary)
            
            if stripes >= 3:  # 要求至少3条条纹
                return True, box
        
        return False, None
        
    def analyze_stripes(self, binary_roi):
        """分析条纹特征"""
        # 计算水平投影
        projection = np.sum(binary_roi, axis=1)
        
        # 使用阈值检测条纹
        threshold = np.mean(projection) * 0.5
        stripes = 0
        in_stripe = False
        
        for value in projection:
            if value > threshold and not in_stripe:
                stripes += 1
                in_stripe = True
            elif value <= threshold:
                in_stripe = False
                
        return stripes
        
    def detect_traffic_light(self, image):
        """检测红绿灯"""
        processed = self.preprocess_image(image)
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(processed, cv2.COLOR_BGR2HSV)
        
        # 定义颜色范围
        red_lower1 = np.array([0, 100, 100])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([170, 100, 100])
        red_upper2 = np.array([180, 255, 255])
        
        yellow_lower = np.array([20, 100, 100])
        yellow_upper = np.array([30, 255, 255])
        
        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])
        
        # 创建颜色掩码
        red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
        red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        
        yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
        green_mask = cv2.inRange(hsv, green_lower, green_upper)
        
        # 形态学操作
        kernel = np.ones((3,3), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
        yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # 检测红色
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        red_circles = self.analyze_contours(red_contours, cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY))
        
        # 检测黄色
        yellow_contours, _ = cv2.findContours(yellow_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        yellow_circles = self.analyze_contours(yellow_contours, cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY))
        
        # 检测绿色
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_circles = self.analyze_contours(green_contours, cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY))
        
        # 优先检测红色
        if red_circles:
            return 2, red_circles[0]  # 红灯
        if yellow_circles:
            return 1, yellow_circles[0]  # 黄灯
        if green_circles:
            return 0, green_circles[0]  # 绿灯
            
        return 0, None  # 默认绿灯
        
    def analyze_contours(self, contours, gray):
        """分析轮廓，找出可能的交通灯"""
        circles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_contour_area:
                continue
                
            # 计算圆形度
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            
            if circularity < self.circle_threshold:
                continue
                
            # 获取轮廓中心点
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # 检查中心点亮度
                if gray[cy, cx] < self.light_threshold:
                    continue
                    
                # 检查位置（通常在图像上半部分）
                if cy > gray.shape[0] * 0.7:  # 如果在下半部分，可能是误检
                    continue
                    
                circles.append((cx, cy, int(np.sqrt(area/np.pi))))
                
        return circles
        
    def visualize_results(self, image, cross_detected, cross_rect, light_state, light_circle):
        """可视化检测结果"""
        display = image.copy()
        
        # 绘制斑马线检测结果
        if cross_detected and cross_rect is not None:
            cv2.drawContours(display, [cross_rect], 0, (0, 255, 0), 2)
            cv2.putText(display, "Crosswalk", (cross_rect[0][0], cross_rect[0][1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 绘制红绿灯检测结果
        if light_circle is not None:
            cx, cy, r = light_circle
            color = (0, 0, 255) if light_state == 2 else \
                   (0, 255, 255) if light_state == 1 else \
                   (0, 255, 0)
            cv2.circle(display, (cx, cy), r, color, 2)
            text = "Red Light" if light_state == 2 else \
                  "Yellow Light" if light_state == 1 else \
                  "Green Light"
            cv2.putText(display, text, (cx-r, cy-r-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return display

class DetectorTester:
    def __init__(self):
        self.detector = TrafficDetector()
        self.test_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "tests": []
        }
        
        # 创建测试结果目录
        if not os.path.exists("test_results"):
            os.makedirs("test_results")
        
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
        while cap.isOpened() and frame_count < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 检测斑马线和红绿灯
            cross_detected, cross_rect = self.detector.detect_crosswalk(frame)
            light_state, light_circle = self.detector.detect_traffic_light(frame)
            
            # 可视化结果
            result_frame = self.detector.visualize_results(
                frame, cross_detected, cross_rect, light_state, light_circle
            )
            
            # 显示当前时间
            current_time = frame_count / fps
            cv2.putText(result_frame, f"Time: {current_time:.1f}s", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 保存结果
            frame_result = {
                "frame_number": frame_count,
                "time": current_time,
                "cross_detected": cross_detected,
                "light_state": light_state,
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
                
        cap.release()
        cv2.destroyAllWindows()
        
        # 保存测试结果
        self.test_results["tests"].append(self.current_test)
        self.save_test_results()
        
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
    tester = DetectorTester()
    
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