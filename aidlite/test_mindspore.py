import cv2
import remi
import mmkv

import sys
import numpy as np

from cvs import *
import aidlite_gpu
aidlite=aidlite_gpu.aidlite(2)

# category labels for deeplabv3_257_mv_gpu.tflite
labels = [ "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "dining table", "dog", "horse", "motorbike", "person", "potted plant", "sheep", "sofa", "train", "tv" ]

def transfer(image, mask):

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = mask

    alpha = 0.7
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

    return dst

w=257
h=257
input_shape=[w,h]
inShape =[1 * w * h *3*4,]
outShape= [1 * w*h*21*4,]
model_path="models/segment_model.ms"
print('hnn gpu mod:',aidlite.ANNModel(model_path,inShape,outShape,4,0))

camid=1
cap=cvs.VideoCapture(camid)

while True:
    t0 = time.time()
    frame=cap.read()
    t1 = time.time()
    if frame is None:
        print('None')
        continue
    if camid==1:
        # frame=cv2.resize(frame,(720,1080))
        frame=cv2.flip(frame,1)
    # frame=cv2.imread('sxg.jpg')   
    
    roi = frame[0:480,80:560] # row0:row1, col0:col1
    # img = cv2.resize(roi,(w,w))
    
    img =cv2.resize(frame,(w,w))
    # input = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    input =img.astype(np.float32)

    input = (input-127.5) / 127.5
    # input=np.transpose(input,(2,0,1))
    print('input',input.shape)


    print('hnn: start set')
    t3 = time.time()
    aidlite.setInput_Float32(input)
    start_time = time.time()
    print('hnn: start invoke')
    t4 = time.time()
    aidlite.invoke()
    
    # t = (time.time() - start_time)
    # # print('elapsed_ms invoke:',t*1000)
    # lbs = 'Fps: '+ str(int(1/t))+" ~~ Time:"+str(t*1000) +"ms"
    # cvs.setLbs(lbs)
    t5 = time.time()
    print('hnn: start get')
    pred_0 = aidlite.getOutput_Float32(0)
    t6 = time.time()
    
    # pred_1 = tflite.t_getTensor_Fp32(1)
    
    
    pred0=(pred_0).reshape(21,w,h)
    t7 = time.time()
    # pred0=np.transpose(pred0,(2,0,1))
    # print('pred:',pred0)


    # find the highest-probability class for each pixel (along axis 2)
    # out = np.apply_along_axis(np.argmax,0,pred0)
    out = np.argmax(pred0, axis=0)
    print('========',w,h,out.shape)
    # print(out)
    t8 = time.time()
      
    # set pixels with likeliest class == person to 255
    pers_idx = labels.index("person")
    print('pers_idx',pers_idx)
    person = np.where(out == pers_idx, 255, 0).astype(np.uint8)
    t9 = time.time()
    # print('person',person)

    dst=transfer(frame,person)
    
    t10 = time.time()

    cvs.imshow(dst)
    t8 = time.time()
    print('cap', t1-t0, 'pre',t3-t1,'setin',t4-t3,'invoke',t5-t4,'getout',t6-t5,'reshape',t7-t6,'prob',t8-t7,'where',t9-t8,'trans',t10-t9,'all',t10-t0)
