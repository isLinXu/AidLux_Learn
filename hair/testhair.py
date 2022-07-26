import cv2
import remi
import mmkv


import sys
import numpy as np

from cvs import *
import aidlite_gpu
aidlite=aidlite_gpu.aidlite(1)


def transfer(image, mask):
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = mask

    alpha = 0.7
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)
    return dst

w=256
h=256
input_shape=[w,h]
inShape =[1 * w * h *3,]
outShape= [1 * w*h,w*h]
model_path="models/segmentation.tnn"
print('gpu:',aidlite.FAST_ANNModel(model_path,inShape,outShape,4,0))

camid=1
cap=cvs.VideoCapture(camid)

while True:
    
    frame=cap.read()
    if frame is None:
        # print('None')
        continue
    if camid==1:
        frame=cv2.flip(frame,1)
    t0=time.time()    
    img =cv2.resize(frame,(w,w))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t1=time.time()
    print('tnn: start set')
    aidlite.setInput_Int8(img,w,w)
    t2 = time.time()
    print('tnn: start invoke')
    aidlite.invoke()
    
    t3 = time.time()
    # print('elapsed_ms invoke:',t*1000)
    # lbs = 'Fps: '+ str(int(1/t))+" ~~ Time:"+str(t*1000) +"ms"
    # cvs.setLbs(lbs)
    # print('tnn: start get')
    pred_0 = aidlite.getOutput_Float32(0)
    pred_1 = aidlite.getOutput_Float32(1)
    print('pred:',pred_0.shape,pred_1.shape)
    t4 = time.time()
    
    pred0=(pred_0).reshape(w,h)
    pred1=(pred_1).reshape(w,h)

     
    back=((pred0)).copy()
    front=((pred1)).copy()
    t5 = time.time()
    mask=front-back

    mask[mask>0]=255
    mask[mask<0]=0

    dst=transfer(frame,mask)
    print('preProcess:%f===setInpu:%f===invoke:%f===getOutput:%f===postProcess:%f' %(t1-t0,t2-t1,t3-t2,t4-t3,t5-t4))
    print('tot', t5-t0)
    cvs.imshow(dst)


