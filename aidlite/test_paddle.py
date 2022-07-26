import cv2
import remi
import mmkv


import sys
import numpy as np

from cvs import *
import aidlite_gpu
aidlite=aidlite_gpu.aidlite()


def transfer(image, mask):


    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = mask

    alpha = 0.7
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

    return dst

w=513
h=513
input_shape=[w,h]
inShape =[1,3,513,513]
# inShape =[1*3*513*513*4]
outShape= [1 * w*h*8,]
model_path="models/model.nb"

camid=1
cap=cvs.VideoCapture(camid)

print('bnn gpu mod:',aidlite.ANNModel(model_path,inShape,outShape,4,0,0))

while True:

# for i in range(5):    
    frame=cvs.read()
    
    if frame is None:
        print('======none')
        continue
    print('======shape', frame.shape)
    if camid==1:
        # frame=cv2.resize(frame,(720,1080))
        frame=cv2.flip(frame,1)
  
    img =cv2.resize(frame,(w,h))

    input =img.astype(np.float32)

    input=np.transpose(input,(2,0,1))

    print('input',input.shape)


    print('bnn: start set')
    
    
    aidlite.setInput_Float32(input)
    print('=========')
    
    start_time = time.time()
    print('bnn: start invoke')
    
    aidlite.invoke()
    print('invoke end')
    t = (time.time() - start_time)
    print('elapsed_ms invoke:',t*1000)
    lbs = 'Fps: '+ str(int(1/t))+" ~~ Time:"+str(t*1000) +"ms"
    cvs.setLbs(lbs)
    
    print('bnn: start get')
    pred_0 = aidlite.getOutput_Int64(0)
    print(pred_0.shape)
    
    # pred_1 = tflite.t_getTensor_Fp32(1)
    
    pred0=(pred_0).reshape(w,h)
    
    # pred0=np.transpose(pred0,(2,0,1))
    print('pred:',pred0.shape)

    print('result',pred0[256][257])


    person = np.where(pred0==1, 255, 0).astype(np.uint8)
    # print('person',person)

    dst=transfer(frame,person)

    cvs.imshow(dst)


