#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode:
    def __init__(self):
        rospy.init_node('camera_node', anonymous=False)
        
        # Publisher: sends images to topic /camera/rgb
        self.image_pub = rospy.Publisher('/camera/rgb', Image, queue_size=2)
        self.bridge = CvBridge()
        
        # Open camera
        self.cap = cv2.VideoCapture(1)  # change index if needed
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        if not self.cap.isOpened():
            rospy.logerr(f"Cannot open camera index 1")
            raise SystemExit
            
        rospy.loginfo("Camera node started, publishing to /camera/rgb")
        
    def run(self):
        rate = rospy.Rate(30)  # 30 FPS
        
        while not rospy.is_shutdown():
            ret, frame = self.cap.read()
            if not ret:
                rospy.logwarn("Failed to grab frame")
                continue
                
            # Flip horizontally (like selfie view)
            frame = cv2.flip(frame, 1)
            
            # Convert OpenCV image to ROS Image message
            ros_image = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            
            # Publish
            self.image_pub.publish(ros_image)
            rate.sleep()
            
    def __del__(self):
        if hasattr(self, 'cap'):
            self.cap.release()

if __name__ == '__main__':
    try:
        node = CameraNode()
        node.run()
    except rospy.ROSInterruptException:
        pass