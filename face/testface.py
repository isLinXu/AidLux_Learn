import cv2
# import tensorflow as tf

import sys
import numpy as np
from blazeface import *
from cvs import *
import aidlite_gpu
aidlite=aidlite_gpu.aidlite(1)
print('aidlite init successed')

        
def plot_detections(img, detections, with_keypoints=True):
        output_img = img
        print(img.shape)

        print("Found %d faces" % len(detections))
        for i in range(len(detections)):
            ymin = detections[i][ 0] * img.shape[0]
            xmin = detections[i][ 1] * img.shape[1] 
            ymax = detections[i][ 2] * img.shape[0]
            xmax = detections[i][ 3] * img.shape[1]
            w=int(xmax-xmin)
            h=int(ymax-ymin)
            if w<h:
                xmin=xmin-(h-w)/3.
                xmax=xmax+(h-w)/3.
            else :
                ymin=ymin-(w-h)/3.
                ymax=ymax+(w-h)/3.
            
            p1 = (int(xmin),int(ymin))
            p2 = (int(xmax),int(ymax))
            print(p1,p2)
            cv2.rectangle(output_img, p1, p2, (0,255,255),2,1)
            cv2.putText(output_img, "Face found! ", (p1[0]+10, p2[1]-10),cv2.FONT_ITALIC, 1, (0, 255, 129), 2)

            if with_keypoints:
                for k in range(6):
                    kp_x = int(detections[i, 4 + k*2    ] * img.shape[1])
                    kp_y = int(detections[i, 4 + k*2 + 1] * img.shape[0])
                    cv2.circle(output_img,(kp_x,kp_y),4,(0,255,255),4)
        # cv2.imwrite("output.png",output_img)

        return output_img
        
input_shape=[128,128]
inShape =[1 * 128 * 128 *3*4,]
outShape= [1 * 896*16*4,1*896*1*4]
model_path="models/face_detection_front.tflite"
print('gpu:',aidlite.FAST_ANNModel(model_path,inShape,outShape,4,0))
# interpreter = tf.lite.Interpreter(model_path="face_detection_front.tflite")
# interpreter.allocate_tensors()

# Get input and output tensors.
# input_details = interpreter.get_input_details()
# output_details = interpreter.get_output_details()

# Test model on random input data.
# input_shape = input_details[0]['shape']
# input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
# interpreter.set_tensor(input_details[0]['index'], input_data)

# interpreter.invoke()

# The function `get_tensor()` returns a copy of the tensor data.
# Use `tensor()` in order to get a pointer to the tensor.
# input_data = interpreter.get_tensor(input_details[0]['index'])
# output_data = interpreter.get_tensor(output_details[0]['index'])
# print(input_data.shape)
# print(output_data.shape)
# fname="star.jpeg"
anchors = np.load('models/anchors.npy').astype(np.float32)
# img = cv2.imread(fname)
camid=1
cap=cvs.VideoCapture(camid)
while True:
    
    frame=cap.read()
    if frame is None:
        continue
    if camid==1:
        # frame=cv2.resize(frame,(720,1080))
        frame=cv2.flip(frame,1)
        
    img =cv2.resize(frame,(128,128))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = img / 127.5 - 1.0

    
    # interpreter.set_tensor(input_details[0]['index'], img[np.newaxis,:,:,:])
    aidlite.setTensor_Fp32(img,input_shape[1],input_shape[1])
    start_time = time.time()
    aidlite.invoke()
    
    t = (time.time() - start_time)
    # print('elapsed_ms invoke:',t*1000)
    lbs = 'Fps: '+ str(int(1/t))+" ~~ Time:"+str(t*1000) +"ms"
    cvs.setLbs(lbs)    
    
    raw_boxes = aidlite.getTensor_Fp32(0)
    classificators = aidlite.getTensor_Fp32(1)
    print(raw_boxes.shape, classificators.shape)

    detections = blazeface(raw_boxes, classificators, anchors)
    out=plot_detections(frame, detections[0])

    cvs.imshow(out)
    # sleep(10)
    

