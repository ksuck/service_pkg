#!/usr/bin/env python3


from __future__ import print_function
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

#topic
#from std_msgs.msg import String, Int32 , Float64

#service
from service_pkg.srv import hand_listener, hand_listenerResponse

import cv2
import numpy as np
import os

import mediapipe as mp

from ultralytics import YOLO

class hand_pose:
    def __init__(self, cam_color, cam_depth, model_yolo = '', revers_cam = False, ros_rate = 500):
        #ชื่อ node
        rospy.init_node('hand_listener_server')
        

        rospy.Subscriber(cam_color, Image, self.rgb_callback)
        rospy.Subscriber(cam_depth, Image, self.depth_callback)

        #สั่งให้ถ่ายรูป
        #ที่อยู่ไฟล์
        python_file = os.path.abspath(__file__)

        # หาที่อยู่ของโฟลเดอร์
        self.current_dir = os.path.dirname(python_file)

    
        #ตั้งค่า mediapipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.bridge = CvBridge()
    

        #ros wait camera
        self.sucess = False
        self.direction = 'None'

        #กลับด้านกล้อง
        self.debug_revers = revers_cam

        #ความเร็วการทำงานกล้อง
        self.rate = rospy.Rate(ros_rate)

        self.model = YOLO(model_yolo)
        


    def rgb_callback(self, data):
        try:
            frame_rgb = self.bridge.imgmsg_to_cv2(data, "bgr8")
            #self.lock.acquire()
            self.frame_rgb = frame_rgb
            if self.debug_revers == True:
                self.frame_rgb = cv2.flip(self.frame_rgb, 1)

            #self.lock.release()
        except Exception as e:
            rospy.logerr(e)

    def depth_callback(self, data):
        try:
            frame_depth = self.bridge.imgmsg_to_cv2(data, desired_encoding="passthrough")
            #self.lock.acquire()
            self.frame_depth = frame_depth
            if self.debug_revers == True:
                self.frame_depth = cv2.flip(self.frame_depth, 1)

            #self.lock.release()
        except Exception as e:
            rospy.logerr(e)



    def main_loop(self , req):
        while not rospy.is_shutdown():
            
            if hasattr(self, 'frame_rgb') and hasattr(self, 'frame_depth'):
                #try:
               
                    results = self.model.track(self.frame_rgb , persist=False, verbose=False)#จำกัดจำนวน [0]
                    h, w, _ = self.frame_rgb.shape
                    frame_copy = self.frame_rgb.copy()

                    
                    if results[0].boxes.id is not None:
                        #id
                        track_ids = results[0].boxes.id.int().cpu()

                        #box
                        boxes = results[0].boxes.xyxy.cpu()
                      
                        #keypoint ตำแหน่งกระดูก
                        result_keypoint = results[0].keypoints.xyn.cpu().numpy()

                    #หาตรงกลางตำแหน่งกระดูกช่วงอก
                        btw_distance = []
                        near_id = 0
                        for res_key , id  in zip(result_keypoint,track_ids):
                            
                            px1 = res_key[5][0] * w
                            py1 = res_key[5][1] * h

                           
                            px2 = res_key[6][0] * w
                            py2 = res_key[6][1] * h

                      
                            if px1 != '' or px2 != '':
                                if px1 != 0 and px2 != 0:
                                    ctx_p = (px1 + px2) /2
                                    cty_p = (py1 + py2) /2

                                    #print(ctx_p,cty_p)

                                    cv2.putText(frame_copy, f"{id}", (abs(int(ctx_p)), abs(int(cty_p))), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                                    cv2.circle(frame_copy , (abs(int(ctx_p)), abs(int(cty_p))), 5, (255, 0, 255), cv2.FILLED)
                                    
                                    if (int(ctx_p) >= 0 and int(ctx_p) <= (w-1)) or int(cty_p) >= 0 and int(cty_p) <= (h-1):

                                        dist =  self.frame_depth[int(cty_p) , int(ctx_p)] #mm


                                        #print(dist)
                                        if dist < 1000:
                                            #print(dist)
                                            #print(int(id),' ',dist)
                                            btw_distance.append((int(id),dist))

                            #หาคนที่ใกล้ที่สุดมาใช้
                            if len(btw_distance) > 0:
                                #print(btw_distance)
                                #หาค่าที่น้อยที่สุด
                                min_value = min(pair[1] for pair in btw_distance)
                                #print(min_value)

                                #เอา id มาใช้เพื่อกำหนด box มา crop
                                for index, item in enumerate(btw_distance):
                                    if item[1] == min_value:
                                        near_id = item[0]
                                        #print(min_value , ' : ', item[1])

                    cropped_image = np.ones((320, 320, 3), dtype=np.uint8) * 255
                    #เอาขนาดคนมาใช้คนที่ใกล้
                    if near_id > 0:
                        print(near_id)
                        x1, y1, x2, y2 = boxes[near_id - 1]
                        x1, y1, x2, y2 = int(x1+10), int(y1+10), int(x2-10), int(y2-10)
                        cropped_image = frame_copy[y1:y2, x1:x2]
                           
                        
                    #hand detection
                        hand_detec = self.hands.process(cropped_image)
                        if  hand_detec.multi_hand_landmarks:
                            for hand_landmarks in hand_detec.multi_hand_landmarks:
                                self.mp_drawing.draw_landmarks(
                                    cropped_image,
                                    hand_landmarks,
                                    self.mp_hands.HAND_CONNECTIONS,
                                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                                    self.mp_drawing_styles.get_default_hand_connections_style()
                                )

                                hand_8x = hand_landmarks.landmark[8].x * w
                                hand_8y = hand_landmarks.landmark[8].y * h

                                hand_0x = hand_landmarks.landmark[0].x * w
                                hand_0y = hand_landmarks.landmark[0].y * h

                                if (hand_8x > hand_0x) and hand_8y  < hand_0y:
                                    self.direction = "left"
                                    self.sucess = True

                                if (hand_8x  < hand_0x) and hand_8y  < hand_0y:
                                    self.direction = "right"
                                    self.sucess = True

                                #cv2.imwrite(self.current_dir,frame_)
                                #cv2.imwrite(self.current_dir,frame_copy)


                    #เช็คค่า server ที่ต้องทำการส่ง
                    if self.direction != 'None':
                        print('Direction : ',self.direction)
                        print('Sucess : ',self.sucess)

                        cv2.imwrite(os.path.join(self.current_dir,'human_pose.jpg'),frame_)
                        cv2.imwrite(os.path.join(self.current_dir,'hand.jpg'),frame_copy)

                        server_response = hand_listenerResponse(self.direction , self.sucess)
                        server_response.success =  self.sucess
                        server_response.direction = self.direction

                        return server_response
                        
                    else:
                        self.direction = "None"
                        self.sucess = False

                        print('Direction : ',self.direction)
                        print('Sucess : ',self.sucess)

                    frame_ = results[0].plot()
                    
                    #cv2.imshow("crop",cropped_image)
                    #cv2.imshow("Image",frame_)
                    #cv2.imshow("RGB", self.frame_rgb)
                    #cv2.imshow("RGB_copy", frame_copy)
                    
                    
                    cv2.waitKey(1)
                #except Exception as e:
                #    rospy.logerr(e)

                    self.rate.sleep()
            

    
    def run(self):
        print("start")
        s = rospy.Service('hand_listener_call', hand_listener, self.main_loop)
        
        
        rospy.spin()




if __name__ == '__main__':
    

    #ที่อยู่ topic กล้อง
    topic_camera_color = '/camera/color/image_raw'
    topic_camera_depth = '/camera/depth/image_raw'

    #ที่อยู่ url Ai 
    yolov8_pose = 'buildnaja/src/Autonomous_Luggage_Companion/all_ai/src/yolov8n-pose.pt'
    
    build_naja = hand_pose(topic_camera_color, topic_camera_depth, yolov8_pose)
    build_naja.run()

    

  

    


  

    