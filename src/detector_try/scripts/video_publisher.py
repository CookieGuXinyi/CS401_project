#!/usr/bin/env python3
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os

class VideoPublisher:
    def __init__(self):
        rospy.init_node('video_publisher', anonymous=True)
        
        # 获取参数
        self.video_path = rospy.get_param('~video_path')
        self.loop = rospy.get_param('~loop', True)
        self.rate = rospy.get_param('~rate', 1.0)
        
        # 检查视频文件是否存在
        if not os.path.exists(self.video_path):
            rospy.logerr(f"视频文件不存在: {self.video_path}")
            rospy.signal_shutdown("Video file not found")
            return
            
        # 初始化发布者
        self.image_pub = rospy.Publisher('/camera/image_raw', Image, queue_size=10)
        
        # 初始化视频捕获
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            rospy.logerr("无法打开视频文件")
            rospy.signal_shutdown("Cannot open video file")
            return
            
        # 初始化转换器
        self.bridge = CvBridge()
        
        # 设置发布频率
        self.rate = rospy.Rate(self.rate)
        
        rospy.loginfo(f"开始发布视频: {self.video_path}")
        
    def run(self):
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            
            if not ret:
                if self.loop:
                    rospy.loginfo("视频播放完毕，重新开始")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                else:
                    rospy.loginfo("视频播放完毕")
                    break
            
            try:
                # 转换图像格式并发布
                msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
                self.image_pub.publish(msg)
            except Exception as e:
                rospy.logerr(f"发布图像时出错: {str(e)}")
            
            self.rate.sleep()
        
        # 清理资源
        self.cap.release()

def main():
    try:
        publisher = VideoPublisher()
        publisher.run()
    except rospy.ROSInterruptException:
        pass

if __name__ == '__main__':
    main() 