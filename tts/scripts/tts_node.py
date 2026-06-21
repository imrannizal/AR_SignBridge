#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import pyttsx3

class TTSNode:
    def __init__(self):
        rospy.init_node('tts_node')
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 150)  # Speed
        rospy.Subscriber('/handsign/speak', String, self.speak)
        rospy.loginfo("TTS ready")
    
    def speak(self, msg):
        text = msg.data.replace('_', ' ')
        rospy.loginfo(f"Speaking: {text}")
        self.engine.say(text)
        self.engine.runAndWait()

if __name__ == '__main__':
    TTSNode()
    rospy.spin()