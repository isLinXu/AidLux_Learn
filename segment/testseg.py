# import cv2
# import tensorflow as tf

import sys
import numpy as np
# from blazeface import *
from cvs import *
import aidlite_gpu
aidlite=aidlite_gpu.aidlite(1)
back_img_path=('res/dock_vbig.jpeg','res/taj_vbig.jpg','/home/cvs/test.jpg','/home/cvs/1.jpg','/home/cvs/2.jpeg','/home/cvs/3.jpg','/home/cvs/4.jpeg')


def find_max_region(mask_sel):
    _,contours,hierarchy = cv2.findContours(mask_sel,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
 
    #找到最大区域并填充 
    area = []
 
    for j in range(len(contours)):
        area.append(cv2.contourArea(contours[j]))
 
    max_idx = np.argmax(area)
 
    # max_area = cv2.contourArea(contours[max_idx])
    
    roi = cv2.boundingRect(contours[max_idx])
    
    
    # cv2.drawContours(img,[contours[max_idx]],-1,(0,0,255),3) 
    
    print ('roi',roi)
    
    return roi


# def seamlessclone(source, bgd, mask,tgt_size):
   
#     # Convert images to UINT8 (0-255)
#     src=np.uint8(source*255.0)
#     dst = np.uint8(bgd*255.0)
#     msk=np.uint8(mask*255.0)

#     # Dilate the mask
#     kernel = np.ones((7,7),np.uint8)
#     msk = cv2.dilate(msk,kernel,iterations = 1)

 
#     # Convert images to BGR format
#     src = cv2.cvtColor(src, cv2.COLOR_RGB2BGR)
#     dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)

#     # Clone size
#     clone_size=tgt_size-2

#     # Resize images
#     src = cv2.resize(src, (clone_size,clone_size),interpolation = cv2.INTER_LINEAR)
#     msk = cv2.resize(msk, (clone_size,clone_size),interpolation = cv2.INTER_LINEAR)

#     # Find contours of mask ROI
#     _,contours, hierarchy = cv2.findContours(msk, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     largest = max(contours, key = cv2.contourArea)   

#     # Find ROI co-ordinates
#     (x,y,w,h) = cv2.boundingRect(largest)
#     X = x+w//2
#     Y = clone_size-h//2
    
#     # Get ROI center
#     center = (X,Y)
#     #print(X+w//2,Y+h//2)

#     # Seamless cloning
#     clone = cv2.seamlessClone(src, dst, msk, center, cv2.NORMAL_CLONE)
#     clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
  
#     return clone

def seamlessclone(input,bgd, mask,w,h):

    
    bgnd_mat=cv2.resize(bgd,(w, h))

    mask = np.uint8(mask*255.0)
    
    _,mask=cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    
    kernel = np.ones((9,9),np.uint8)
    
    mask=cv2.dilate(mask, None,iterations=1);
    
    mask_mat = cv2.resize(mask, (w, h),interpolation=cv2.INTER_NEAREST)
    input_mat = cv2.resize(input, (w,h),interpolation=cv2.INTER_NEAREST)

    roi=find_max_region(mask_mat)
    mask_mat=cv2.cvtColor(mask_mat,cv2.COLOR_GRAY2BGR)

    input_mat=input_mat[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]] 
    mask_mat=mask_mat[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]      
    mask_mat[mask_mat>0]=255
    
    width, height, channels = bgnd_mat.shape
    center =( width-input_mat.shape[1]//2, height- input_mat.shape[0]// 2)
    # center =(height//2, width// 2)
    # center =(input_mat.shape[1]//2, input_mat.shape[0]// 2)

    normal_clone = cv2.seamlessClone(input_mat, bgnd_mat, mask_mat, center, cv2.NORMAL_CLONE)
    
    # center =( (height)//2, (width// 2))

    
    # mask_mat=cv2.cvtColor(mask_mat,cv2.COLOR_BGR2GRAY)
    # mask_mat=cv2.resize(normal_clone, (mask_mat.shape[1]*2,mask_mat.shape[0]*2))
    # _,mask_mat=cv2.threshold(mask_mat,0,255,cv2.THRESH_BINARY)
    
    
    maskbig=mask#cv2.dilate(mask, kernel,iterations=1);
    # masksmall=cv2.erode(mask, kernel,iterations=1);
    masksmall=mask
    # maskcircle=maskbig-masksmall
    
    ww=bgd.shape[0]
    hh=bgd.shape[1]
    
    normal_clone=cv2.resize(normal_clone, (ww,hh))
    
    # normal_clone = np.uint8(normal_clone)
    # maskbig=cv2.resize(maskbig, (ww,hh),interpolation=cv2.INTER_NEAREST)
    # masksmall=cv2.resize(masksmall, (ww,hh),interpolation=cv2.INTER_NEAREST)
    # maskcircle=cv2.resize(maskcircle, (ww,hh),interpolation=cv2.INTER_NEAREST)
    maskbig_inv=cv2.bitwise_not(mask)
    
    imgbig=cv2.bitwise_and(bgd,bgd,mask = maskbig_inv)
    # imgcircle=cv2.bitwise_and(normal_clone,normal_clone,mask = maskcircle)
    # imgcircle1=cv2.bitwise_and(bgd,bgd,mask = maskcircle)
    # imgcircle2=cv2.bitwise_and(input,input,mask = maskcircle)
    # imgcircle=(imgcircle1+imgcircle+imgcircle2)/3
    
    imgsmall=cv2.bitwise_and(normal_clone,normal_clone,mask = mask)
    
    normal=imgbig+imgsmall


    return normal_clone
    

def transfer_background_big(input,bgd, mask):

    


    mask = np.uint8(mask*255.0)
    
    _,mask=cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    
    mask=cv2.dilate(mask, None,iterations=1);
    
 
    kernel = np.ones((9,9),np.uint8)
    masksmall=cv2.erode(mask, kernel,iterations=2);
    # maskcircle=maskbig-masksmall
    
    ww=bgd.shape[0]
    hh=bgd.shape[1]
    

    maskbig_inv=cv2.bitwise_not(masksmall)
    
    imgbig=cv2.bitwise_and(bgd,bgd,mask = maskbig_inv)

    
    imgsmall=cv2.bitwise_and(input,input,mask = masksmall)
    
    normal_clone=imgbig+imgsmall


    return normal_clone
    



def transfer_background(input_mat,bgnd_mat, mask):

    
    # bgnd_mat=cv2.resize(bgnd_mat,(input_mat.shape[1], input_mat.shape[0]))
    mask = cv2.resize(mask, (mask.shape[1]//2, mask.shape[0]//2)).astype(np.uint8)
    
    _,mask_mat=cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    
    mask_mat=cv2.dilate(mask_mat, None,iterations=2);
    
    mask_mat = cv2.resize(mask_mat, (input_mat.shape[1]//2, input_mat.shape[0]//2),interpolation=cv2.INTER_NEAREST)
    input_mat = cv2.resize(input_mat, (input_mat.shape[1]//2, input_mat.shape[0]//2),interpolation=cv2.INTER_NEAREST)
    #contours, hierarchy = cv2.findContours(mask_mat,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) 
    roi=find_max_region(mask_mat)
    mask_mat=cv2.cvtColor(mask_mat,cv2.COLOR_GRAY2BGR)
    
    input=input_mat[roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]] 
    mask=mask_mat[roi[0]:roi[0]+roi[2],roi[1]:roi[1]+roi[3]] 
    
    
    # input=input_mat[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]] 
    # mask=mask_mat[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]  
    
    mask[mask>0]=255
    
    # mask_mat = mask_mat*255;

    width, height, channels = bgnd_mat.shape
    
    center =( width//2, (height)// 2)
    
    # center =( roi[0]+roi[2]//2, height-roi[3]// 2)
    
    # center =( (height-input.shape[1]//2), (width- input.shape[0]// 2))
    print(center)
    
    # if((center[0]+input.shape[1])>=height or (center[1]+input.shape[0]>=width) ):
        # return bgnd_mat
    
    try:
        normal_clone = cv2.seamlessClone(input, bgnd_mat, mask, center, cv2.NORMAL_CLONE)
    except:
        return bgnd_mat


    return normal_clone
    

def transfer_back(image, mask):


    mask = cv2.resize(mask, (image.shape[1], image.shape[0])).astype(np.uint8)
    
    _,mask=cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    # mask_inv=cv2.dilate(mask_inv,None,iterations=30)
    back=cv2.bitwise_and(image,image,mask = mask_inv)
    blur=cv2.blur(back,(30,30))
    # blur=cv2.dilate(blur,None,iterations=2)
    
    mask=cv2.erode(mask,None,iterations=2)
    # cv2.imshow('erode',erode)
    # dilate=cv2.dilate(erode,None,iterations=2)
    # cv2.imshow('dilate',dilate)

    dst = cv2.bitwise_and(image,image,mask = mask)
    # back[::3]=255
    # back[::1]=0
    # back[::2]=0
    
    # back[0]=255
    
    
    dst=cv2.add(dst,blur)

    return dst

def transfer_back_color(image, mask):


    mask = cv2.resize(mask, (image.shape[1], image.shape[0])).astype(np.uint8)
    
    _,mask=cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)

    mask_inv=cv2.dilate(mask_inv,None,iterations=30)
    back=cv2.bitwise_and(image,image,mask = mask_inv)
    blur=cv2.blur(back,(30,30))
    # blur=cv2.dilate(blur,None,iterations=2)
    
    mask=cv2.erode(mask,None,iterations=2)
    # cv2.imshow('erode',erode)
    # dilate=cv2.dilate(erode,None,iterations=2)
    # cv2.imshow('dilate',dilate)

    dst = cv2.bitwise_and(image,image,mask = mask)
    # back[::3]=255
    # back[::1]=0
    # back[::2]=0
    
    # back[0]=255
    
    
    dst=cv2.add(dst,blur)

    return dst

def transfer(image, mask):
    # mask[mask > 0.5] = 255
    # mask[mask <= 0.5] = 0

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))


    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = mask



    alpha = 0.5
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

    return dst

w=512
h=512
input_shape=[w,h]
inShape =[1 * w * h *3*4,]
outShape= [1 * w*h*2*4,]
model_path="res/portrait_segmentation.tflite"
print('gpu:',aidlite.FAST_ANNModel(model_path,inShape,outShape,4,0))

camid=0
cap=cvs.VideoCapture(camid)
bgnd_mat=cv2.imread(back_img_path[1])
tgt_size=512
bgnd_mat=cv2.resize(bgnd_mat,(tgt_size, tgt_size))
bgd = cv2.cvtColor(bgnd_mat, cv2.COLOR_BGR2RGB)/255.0

while True:
    
    frame=cap.read()
    if frame is None:
        continue
    if camid==1:
        # frame=cv2.resize(frame,(720,1080))
        frame=cv2.flip(frame,1)

        
    img =cv2.resize(frame,(input_shape[0],input_shape[1]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    # img = img / 127.5 - 1.0
    img = img / 255
    
    print ('img',img.shape)

    
    # interpreter.set_tensor(input_details[0]['index'], img[np.newaxis,:,:,:])
    aidlite.setTensor_Fp32(img,input_shape[1],input_shape[1])
    start_time = time.time()
    aidlite.invoke()
    
    t = (time.time() - start_time)
    # print('elapsed_ms invoke:',t*1000)
    lbs = 'Fps: '+ str(int(1/t))+" ~~ Time:"+str(t*1000) +"ms"
    cvs.setLbs(lbs)    
    
    pred = aidlite.getTensor_Fp32(0)
    
    pred0=(pred[::2 ]).reshape(w,h)
    pred1=(pred[1::2]).reshape(w,h)

     
    back=((pred0)).copy()
    front=((pred1)).copy()
    
    front[front<=0]=0
    back[back<=0]=0
    
    
    mask=front-back
    
    mask[mask>0.0]=255
    mask[mask<=0.0]=0
    
    # maxPredictionVal = -0xFF;
    # front[front > 0.5] = 255
    # front[front <= 0.5] = 0

    # back[back > 0.5] = 255
    # back[back <= 0.5] = 0    
    
    # mask=front+back
    
    # print('mask:',mask.shape)
    # mask=cv2.resize(mask,(720,1080))
    # mask=((-pred0)).copy()
    # dst=transfer(frame,mask)
    # mask=((pred1)).copy()
    
    # dst=transfer(frame,mask)
    msk=cv2.GaussianBlur(mask,(7,7),1)
    frame=cv2.resize(frame,(tgt_size,tgt_size))
    dst=transfer_background_big(frame,bgnd_mat,msk)
    # dst=seamlessclone(frame,bgnd_mat,msk,300,300)
    # dst=cv2.GaussianBlur(dst,(5,5),1)
    # msk=cv2.GaussianBlur(mask,(7,7),1)
    # img=cv2.resize(frame, (tgt_size,tgt_size))/255.0
    # msk=cv2.resize(mask, (tgt_size,tgt_size)).reshape((tgt_size,tgt_size,1))    
    
    # dst=seamlessclone(img,bgd,msk,tgt_size)

    # cvs.imshow(mask)
    cvs.imshow(dst)
    sleep(1)
   
    

