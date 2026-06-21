#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import speech_recognition as sr

class STTNode:
    def __init__(self):
        rospy.init_node('stt_node')
        self.pub = rospy.Publisher('/handsign/voice_command', String, queue_size=2)
        self.recognizer = sr.Recognizer()
        rospy.loginfo("STT ready")
    
    def run(self):
        while not rospy.is_shutdown():
            try:
                with sr.Microphone() as source:
                    rospy.loginfo("Listening...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=5, phrase_time_limit=2)
                    
                    text = self.recognizer.recognize_google(audio).lower()
                    rospy.loginfo(f"Heard: {text}")
                    self.pub.publish(String(data=text))
                    
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                rospy.loginfo("Didn't understand")
            except sr.RequestError:
                rospy.loginfo("STT service error")
            except Exception as e:
                rospy.logerr(f"Error: {e}")

if __name__ == '__main__':
    stt = STTNode()
    stt.run()