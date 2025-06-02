#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import os

class VideoRecorder:
    def __init__(self):
        rospy.init_node('video_recorder', anonymous=True)
        
        # 初始化订阅者
        self.image_sub = rospy.Subscriber('/detector_try/processed_image', Image, self.image_callback)
        
        # 初始化变量
        self.bridge = CvBridge()
        self.recording = False
        self.video_writer = None
        
        # 创建保存目录
        self.save_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_videos')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        rospy.loginfo("视频录制器已初始化")
        rospy.loginfo("按 'r' 开始/停止录制")
        rospy.loginfo("按 'q' 退出")
    
    def image_callback(self, msg):
        try:
            # 转换ROS图像消息为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 显示图像
            cv2.imshow('Recording', cv_image)
            
            # 处理键盘输入
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.toggle_recording()
            elif key == ord('q'):
                self.stop_recording()
                rospy.signal_shutdown('User requested shutdown')
            
            # 如果正在录制，写入视频
            if self.recording and self.video_writer is not None:
                self.video_writer.write(cv_image)
                
        except Exception as e:
            rospy.logerr(f"图像处理错误: {str(e)}")
    
    def toggle_recording(self):
        if not self.recording:
            # 开始录制
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(self.save_dir, f'test_video_{timestamp}.avi')
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.video_writer = cv2.VideoWriter(filename, fourcc, 20.0, (640, 480))
            self.recording = True
            rospy.loginfo(f"开始录制视频: {filename}")
        else:
            # 停止录制
            self.stop_recording()
    
    def stop_recording(self):
        if self.recording:
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.recording = False
            rospy.loginfo("停止录制视频")
    
    def cleanup(self):
        self.stop_recording()
        cv2.destroyAllWindows()

def main():
    recorder = VideoRecorder()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        recorder.cleanup()

if __name__ == '__main__':
    main() 