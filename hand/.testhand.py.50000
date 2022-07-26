import cv2
import math
# import tensorflow as tf

import sys
import numpy as np
from blazeface import *
from cvs import *
import aidlite_gpu
aidlite=aidlite_gpu.aidlite()

def preprocess_image_for_tflite32(image, model_image_size=300):
    print(type(image))
    print(image.shape)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (model_image_size, model_image_size))
    image = np.expand_dims(image, axis=0)
    image = (2.0 / 255.0) * image - 1.0
    image = image.astype('float32')

    return image
        
def plot_detections(img, detections, with_keypoints=True):
        output_img = img
        print(img.shape)
        x_min=[0,0]
        x_max=[0,0]
        y_min=[0,0]
        y_max=[0,0]
        hand_nums=len(detections)
        # if hand_nums >2:
        #     hand_nums=2
        print("Found %d hands" % hand_nums)
        if hand_nums >2:
            hand_nums=2        
        for i in range(hand_nums):
            ymin = detections[i][ 0] * img.shape[0]
            xmin = detections[i][ 1] * img.shape[1] 
            ymax = detections[i][ 2] * img.shape[0]
            xmax = detections[i][ 3] * img.shape[1]
            w=int(xmax-xmin)
            h=int(ymax-ymin)
            h=max(h,w)
            h=h*224./128.
            # ymin-=0.08*h
            
            # xmin-=0.25*w
            # xmax=xmin+1.5*w;
            # ymax=ymin+1.0*h;
            
            x=(xmin+xmax)/2.
            y=(ymin+ymax)/2.
            
            xmin=x-h/2.
            xmax=x+h/2.
            ymin=y-h/2.-0.18*h
            ymax=y+h/2.-0.18*h
            
            # if w<h:
            #     xmin=xmin-(h+0.08*h-w)/2
            #     xmax=xmax+(h+0.08*h-w)/2
            #     ymin-=0.08*h
            #     # ymax-=0.08*h
            # else :
            #     ymin=ymin-(w-h)/2
            #     ymax=ymax+(w-h)/2
            
            # h=int(ymax-ymin)
            # ymin-=0.08*h            
            # landmarks_xywh[:, 2:4] += (landmarks_xywh[:, 2:4] * pad_ratio).astype(np.int32) #adding some padding around detection for landmark detection step.
            # landmarks_xywh[:, 1:2] -= (landmarks_xywh[:, 3:4]*0.08).astype(np.int32)
            
            x_min[i]=int(xmin)
            y_min[i]=int(ymin)
            x_max[i]=int(xmax)
            y_max[i]=int(ymax)            
            p1 = (int(xmin),int(ymin))
            p2 = (int(xmax),int(ymax))
            # print(p1,p2)
            cv2.rectangle(output_img, p1, p2, (0,255,255),2,1)
            
            # cv2.putText(output_img, "Face found! ", (p1[0]+10, p2[1]-10),cv2.FONT_ITALIC, 1, (0, 255, 129), 2)

            # if with_keypoints:
            #     for k in range(7):
            #         kp_x = int(detections[i, 4 + k*2    ] * img.shape[1])
            #         kp_y = int(detections[i, 4 + k*2 + 1] * img.shape[0])
            #         cv2.circle(output_img,(kp_x,kp_y),4,(0,255,255),4)

        return x_min,y_min,x_max,y_max
        
def draw_mesh(image, mesh, mark_size=4, line_width=1):
    """Draw the mesh on an image"""
    # The mesh are normalized which means we need to convert it back to fit
    # the image size.
    image_size = image.shape[0]
    mesh = mesh * image_size
    for point in mesh:
        cv2.circle(image, (point[0], point[1]),
                   mark_size, (255, 0, 0), 4)

    # Draw the contours.
    # Eyes
    # left_eye_contour = np.array([mesh[33][0:2],
    #                              mesh[7][0:2],
    #                              mesh[163][0:2],
    #                              mesh[144][0:2],
    #                              mesh[145][0:2],
    #                              mesh[153][0:2],
    #                              mesh[154][0:2],
    #                              mesh[155][0:2],
    #                              mesh[133][0:2],
    #                              mesh[173][0:2],
    #                              mesh[157][0:2],
    #                              mesh[158][0:2],
    #                              mesh[159][0:2],
    #                              mesh[160][0:2],
    #                              mesh[161][0:2],
    #                              mesh[246][0:2], ]).astype(np.int32)
    # right_eye_contour = np.array([mesh[263][0:2],
    #                               mesh[249][0:2],
    #                               mesh[390][0:2],
    #                               mesh[373][0:2],
    #                               mesh[374][0:2],
    #                               mesh[380][0:2],
    #                               mesh[381][0:2],
    #                               mesh[382][0:2],
    #                               mesh[362][0:2],
    #                               mesh[398][0:2],
    #                               mesh[384][0:2],
    #                               mesh[385][0:2],
    #                               mesh[386][0:2],
    #                               mesh[387][0:2],
    #                               mesh[388][0:2],
    #                               mesh[466][0:2]]).astype(np.int32)
    # # Lips
    # cv2.polylines(image, [left_eye_contour, right_eye_contour], False,
    #               (255, 255, 255), line_width, cv2.LINE_AA)
def calc_palm_moment(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    palm_array = np.empty((0, 2), int)

    for index, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark[0] * image_width), image_width - 1)
        landmark_y = min(int(landmark[1] * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        if index == 0:  # 手首1
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 1:  # 手首2
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 5:  # 人差指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 9:  # 中指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 13:  # 薬指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
        if index == 17:  # 小指：付け根
            palm_array = np.append(palm_array, landmark_point, axis=0)
    M = cv2.moments(palm_array)
    cx, cy = 0, 0
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

    return cx, cy

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks):
        landmark_x = min(int(landmark[0] * image_width), image_width - 1)
        landmark_y = min(int(landmark[0] * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv2.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # 外接矩形
        cv2.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 255, 0), 2)

    return image
    
def draw_landmarks(image, cx, cy, landmarks):
    
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # キーポイント
    for index, landmark in enumerate(landmarks):
        # if landmark.visibility < 0 or landmark.presence < 0:
        #     continue

        landmark_x = min(int(landmark[0] * image_width), image_width - 1)
        landmark_y = min(int(landmark[1] * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append((landmark_x, landmark_y))

        if index == 0:  # 手首1
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 手首2
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 親指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 親指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 親指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 5:  # 人差指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 人差指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 人差指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 人差指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 9:  # 中指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 中指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 中指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 中指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 13:  # 薬指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 薬指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 薬指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 薬指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)
        if index == 17:  # 小指：付け根
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 小指：第2関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 小指：第1関節
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 小指：指先
            cv2.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
            cv2.circle(image, (landmark_x, landmark_y), 12, (0, 255, 0), 2)

    # 接続線
    if len(landmark_point) > 0:
        # 親指
        cv2.line(image, landmark_point[2], landmark_point[3], (0, 255, 0), 2)
        cv2.line(image, landmark_point[3], landmark_point[4], (0, 255, 0), 2)

        # 人差指
        cv2.line(image, landmark_point[5], landmark_point[6], (0, 255, 0), 2)
        cv2.line(image, landmark_point[6], landmark_point[7], (0, 255, 0), 2)
        cv2.line(image, landmark_point[7], landmark_point[8], (0, 255, 0), 2)

        # 中指
        cv2.line(image, landmark_point[9], landmark_point[10], (0, 255, 0), 2)
        cv2.line(image, landmark_point[10], landmark_point[11], (0, 255, 0), 2)
        cv2.line(image, landmark_point[11], landmark_point[12], (0, 255, 0), 2)

        # 薬指
        cv2.line(image, landmark_point[13], landmark_point[14], (0, 255, 0), 2)
        cv2.line(image, landmark_point[14], landmark_point[15], (0, 255, 0), 2)
        cv2.line(image, landmark_point[15], landmark_point[16], (0, 255, 0), 2)

        # 小指
        cv2.line(image, landmark_point[17], landmark_point[18], (0, 255, 0), 2)
        cv2.line(image, landmark_point[18], landmark_point[19], (0, 255, 0), 2)
        cv2.line(image, landmark_point[19], landmark_point[20], (0, 255, 0), 2)

        # 手の平
        cv2.line(image, landmark_point[0], landmark_point[1], (0, 255, 0), 2)
        cv2.line(image, landmark_point[1], landmark_point[2], (0, 255, 0), 2)
        cv2.line(image, landmark_point[2], landmark_point[5], (0, 255, 0), 2)
        cv2.line(image, landmark_point[5], landmark_point[9], (0, 255, 0), 2)
        cv2.line(image, landmark_point[9], landmark_point[13], (0, 255, 0), 2)
        cv2.line(image, landmark_point[13], landmark_point[17], (0, 255, 0), 2)
        cv2.line(image, landmark_point[17], landmark_point[0], (0, 255, 0), 2)

    # 重心 + 左右
    if len(landmark_point) > 0:
        # handedness.classification[0].index
        # handedness.classification[0].score

        cv2.circle(image, (cx, cy), 12, (0, 255, 0), 2)
        # cv2.putText(image, handedness.classification[0].label[0],
        #           (cx - 6, cy + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0),
        #           2, cv2.LINE_AA)  # label[0]:一文字目だけ

    return image


input_shape=[128,128]
inShape =[1 * 128 * 128 *3*4,]
outShape= [1 * 896*18*4,1*896*1*4]
model_path="models/palm_detection.tflite"
print('gpu:',aidlite.FAST_ANNModel(model_path,inShape,outShape,4,0))
model_path="models/hand_landmark.tflite"
aidlite.set_g_index(1)
inShape1 =[1 * 224 * 224 *3*4,]
outShape1= [1 * 63*4,1*4,1*4]
print('cpu:',aidlite.FAST_ANNModel(model_path,inShape1,outShape1,4,0))

anchors = np.load('models/anchors.npy').astype(np.float32)
camid=1
cap=cvs.VideoCapture(camid)
bHand=False

x_min=[0,0]
x_max=[0,0]
y_min=[0,0]
y_max=[0,0]

fface=0.0
use_brect=True

while True:
    
    frame=cvs.read()
    if frame is None:
        continue
    if camid==1:
        # frame=cv2.resize(frame,(240,480))
        frame=cv2.flip(frame,1)
        
    start_time = time.time()    
    img = preprocess_image_for_tflite32(frame,128)

    
    # interpreter.set_tensor(input_details[0]['index'], img[np.newaxis,:,:,:])
    if bHand==False:
        aidlite.set_g_index(0)
        aidlite.setTensor_Fp32(img,input_shape[1],input_shape[1])
        
        aidlite.invoke()
        
        # t = (time.time() - start_time)
        # # print('elapsed_ms invoke:',t*1000)
        # lbs = 'Fps: '+ str(int(1/t))+" ~~ Time:"+str(t*1000) +"ms"
        # cvs.setLbs(lbs)    
        
        raw_boxes = aidlite.getTensor_Fp32(0)
        classificators = aidlite.getTensor_Fp32(1)
    
        detections = blazeface(raw_boxes, classificators, anchors)

        x_min,y_min,x_max,y_max=plot_detections(frame, detections[0])
        if len(detections[0])>0 :
            bHand=True
    if bHand:
        hand_nums=len(detections[0])
        if hand_nums>2:
            hand_nums=2
        for i in range(hand_nums):
            
            print(x_min,y_min,x_max,y_max)
            xmin=max(0,x_min[i])
            ymin=max(0,y_min[i])
            xmax=min(frame.shape[1],x_max[i])
            ymax=min(frame.shape[0],y_max[i])
    
            roi_ori=frame[ymin:ymax, xmin:xmax]
            # cvs.imshow(roi)
            roi =preprocess_image_for_tflite32(roi_ori,224)
               
            aidlite.set_g_index(1)
            aidlite.setTensor_Fp32(roi,224,224)
            # start_time = time.time()
            aidlite.invoke()
            mesh = aidlite.getTensor_Fp32(0)
            # ffacetmp = tflite.getTensor_Fp32(1)[0]
            # print('fface:',abs(fface-ffacetmp))
            # if abs(fface - ffacetmp) > 0.5:
            bHand=False
            # fface=ffacetmp
                
            # print('mesh:',mesh.shape)
            mesh = mesh.reshape(21, 3)/224
            cx, cy = calc_palm_moment(roi_ori, mesh)
            draw_landmarks(roi_ori,cx,cy,mesh)
            # brect = calc_bounding_rect(roi_ori, mesh)
            # draw_bounding_rect(use_brect, roi_ori, brect)
            # draw_mesh(roi_ori,mesh)
            frame[ymin:ymax, xmin:xmax]=roi_ori
    
    t = (time.time() - start_time)
    # print('elapsed_ms invoke:',t*1000)
    lbs = 'Fps: '+ str(int(100/t)/100.)+" ~~ Time:"+str(t*1000) +"ms"
    cvs.setLbs(lbs) 
    
    cvs.imshow(frame)
    sleep(1)
    
import apkneed
