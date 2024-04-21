#!/usr/bin/env python3
import rospy
from service_pkg.srv import hand_listener, hand_listenerRequest

def add_two_ints_client():
    rospy.init_node('hand_listener_client')
    rospy.wait_for_service('hand_listener_call')
    
    
    
    add_two_ints = rospy.ServiceProxy('hand_listener_call', hand_listener)

    # สร้าง request object โดยไม่ระบุข้อมูล
    req = hand_listenerRequest()

    resp = add_two_ints(req)

    print(resp)
  
    

if __name__=='__main__':
    add_two_ints_client()
    #print(direction, success)
