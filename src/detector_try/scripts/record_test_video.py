#!/usr/bin/env python3

import cv2
import os
import time
from datetime import datetime

def record_video(output_path, duration=10, fps=30):
    """录制测试视频
    
    Args:
        output_path: 输出视频路径
        duration: 录制时长（秒）
        fps: 帧率
    """
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头")
        return
        
    # 设置摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (640, 480))
    
    print(f"开始录制视频，时长 {duration} 秒...")
    start_time = time.time()
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 显示录制时间
        elapsed = time.time() - start_time
        cv2.putText(frame, f"Recording: {elapsed:.1f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 写入帧
        out.write(frame)
        
        # 显示预览
        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # 释放资源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print(f"视频已保存到: {output_path}")

def main():
    # 创建测试视频目录
    test_videos_dir = os.path.join(os.path.dirname(__file__), "..", "test_videos")
    os.makedirs(test_videos_dir, exist_ok=True)
    
    # 录制不同类型的测试视频
    test_cases = [
        ("crosswalk_front.mp4", "正面视角的斑马线"),
        ("crosswalk_side.mp4", "侧面视角的斑马线"),
        ("traffic_light_red.mp4", "红灯"),
        ("traffic_light_yellow.mp4", "黄灯"),
        ("traffic_light_green.mp4", "绿灯"),
        ("red_light_crosswalk.mp4", "红灯和斑马线")
    ]
    
    for filename, description in test_cases:
        print(f"\n准备录制: {description}")
        print("按回车键开始录制...")
        input()
        
        output_path = os.path.join(test_videos_dir, filename)
        record_video(output_path)
        
        print("\n录制完成！")
        print("1. 继续录制下一个视频")
        print("2. 退出")
        choice = input("请选择 (1/2): ")
        if choice != "1":
            break

if __name__ == "__main__":
    main() 