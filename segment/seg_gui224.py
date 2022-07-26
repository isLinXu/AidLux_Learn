
from cvs import *

import numpy as np

import tflite_gpu
tflite=tflite_gpu.tflite()
back_img_path=('res/dock_vbig.jpeg','res/taj_vbig.jpg','res/sunset_vbig.jpg','/home/cvs/test.jpg','/home/cvs/1.jpg','/home/cvs/2.jpeg','/home/cvs/3.jpg','/home/cvs/4.jpeg')

bgnd_mat=cv2.imread(back_img_path[0])
mod=0

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


def seamlessclone(input,bgd, mask,w,h):

    
    bgnd_mat=cv2.resize(bgd,(w, h))

    mask = np.uint8(mask*255.0)
    
    _,mask=cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    
    kernel = np.ones((7,7),np.uint8)
    
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


    normal_clone = cv2.seamlessClone(input_mat, bgnd_mat, mask_mat, center, cv2.NORMAL_CLONE)
    
    return normal_clone
    

def transfer_background_big(input,bgd, mask):

    mask = np.uint8(mask*255.0)
    
    _,mask=cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    
    mask=cv2.dilate(mask, None,iterations=1);
    
 
    kernel = np.ones((7,7),np.uint8)
    masksmall=cv2.erode(mask, kernel,iterations=2);
    # maskcircle=maskbig-masksmall
    
    ww=bgd.shape[0]
    hh=bgd.shape[1]
    

    maskbig_inv=cv2.bitwise_not(masksmall)
    
    imgbig=cv2.bitwise_and(bgd,bgd,mask = maskbig_inv)

    
    imgsmall=cv2.bitwise_and(input,input,mask = masksmall)
    
    normal_clone=imgbig+imgsmall


    return normal_clone
    

def transfer(image, mask):
    # mask[mask > 0.5] = 255
    # mask[mask <= 0.5] = 0

    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))


    mask_n = np.zeros_like(image)
    mask_n[:, :, 0] = mask



    alpha = 0.3
    beta = (1.0 - alpha)
    dst = cv2.addWeighted(image, alpha, mask_n, beta, 0.0)

    return dst



class MyApp(App):
   
    def __init__(self, *args):
        super(MyApp, self).__init__(*args)
    
    def idle(self):
        self.aidcam.update()
        
    def main(self):
        #creating a container VBox type, vertical (you can use also HBox or Widget)
        main_container = VBox(width=360, height=680, style={'margin':'0px auto'})
        
        self.aidcam = OpencvVideoWidget(self, width=340, height=400)
        self.aidcam.style['margin'] = '10px'
        
        self.aidcam.identifier="myimage_receiver"
        main_container.append(self.aidcam)
        
        self.lbl = Label('点击图片选择你喜欢的虚拟背景：')
        main_container.append(self.lbl)
        
        bottom_container = HBox(width=360, height=230, style={'margin':'0px auto'})
        self.img1 = Image('/res:'+os.getcwd()+'/res/dock_vbig.jpeg', height=100, margin='10px')
        self.img1.onclick.do(self.on_img1_clicked)
        bottom_container.append(self.img1)
        
        self.img2 = Image('/res:'+os.getcwd()+'/res/taj_vbig.jpg', height=100, margin='10px')
        self.img2.onclick.do(self.on_img2_clicked)
        bottom_container.append(self.img2)
        
        self.img3 = Image('/res:'+os.getcwd()+'/res/sunset_vbig.jpg', height=100, margin='10px')
        self.img3.onclick.do(self.on_img3_clicked)
        bottom_container.append(self.img3)
        
        self.bt1 = Button('抠图模式', width=300, height=30, margin='10px')
        self.bt1.onclick.do(self.on_button_pressed1)
        
        # self.bt2 = Button('渲染模式', width=100, height=30, margin='10px')
        # self.bt2.onclick.do(self.on_button_pressed2)        

        self.bt3 = Button('背景模糊', width=300, height=30, margin='10px')
        self.bt3.onclick.do(self.on_button_pressed3) 
        
        main_container.append(bottom_container)
        main_container.append(self.bt1)
        # main_container.append(self.bt2)
        main_container.append(self.bt3)
        

        return main_container
        
    def on_img1_clicked(self, widget):
        global bgnd_mat
        bgnd=cv2.imread(back_img_path[0])
        bgnd_mat=cv2.resize(bgnd,(512, 512))
        global bgnd_matsmall
        bgnd_matsmall=cv2.resize(bgnd_mat,(256,144))
        
    def on_img2_clicked(self, widget):
        global bgnd_mat
        bgnd=cv2.imread(back_img_path[1])
        bgnd_mat=cv2.resize(bgnd,(512, 512))
        global bgnd_matsmall
        bgnd_matsmall=cv2.resize(bgnd_mat,(256,144))
        
    def on_img3_clicked(self, widget):
        global bgnd_mat
        bgnd=cv2.imread(back_img_path[2])       
        bgnd_mat=cv2.resize(bgnd,(512, 512))
        global bgnd_matsmall
        bgnd_matsmall=cv2.resize(bgnd_mat,(256,144))
        
    def on_button_pressed1(self, widget):
        global mod
        mod=0
        
    # def on_button_pressed2(self, widget):
    #     global mod
    #     mod=1   
        
    def on_button_pressed3(self, widget):
        global mod
        mod=2
        
def process():
    cvs.setCustomUI()
    # cap=cvs.VideoCapture(1)
    



    # while True:
    #     sleep(30)
    #     img =cvs.read()

    #     if img is None :
    #         continue

 
    #     cvs.imshow(img)
    w=144
    h=256
    input_shape=[w,h]
    inShape =[1 * w * h *3*4,]
    outShape= [1 * w*h*4*2]
    model_path="res/model_float16_quant.tflite"
    print('gpu:',tflite.NNModel(model_path,inShape,outShape,4,0))
    
    
    ww=512
    hh=512
    input_shape1=[ww,hh]
    inShape1 =[1 * ww * hh *4*4,]
    outShape1= [1 * ww*hh*2*4,]
    model_path="res/hair_segmentation.tflite"
    tflite.set_g_index(1)
    print('gpu1:',tflite.NNModel(model_path,inShape1,outShape1,4,0)) 
    
    
    
    camid=1
    cap=cvs.VideoCapture(camid)
    
    tgt_size=512
    global bgnd_mat
    bgnd_mat=cv2.resize(bgnd_mat,(tgt_size,tgt_size),interpolation=cv2.INTER_NEAREST)
    bgd = cv2.cvtColor(bgnd_mat, cv2.COLOR_BGR2RGB)/255.0
    
    in_tensor = np.zeros((512,512,4),dtype=np.float32)
    masa = np.zeros((512,512),dtype=np.float32)
    # framesamll=np.zeros((h,w),dtype=np.float32)
    # global bgnd_matsmall
    # bgnd_matsmall=cv2.resize(bgnd_mat,(h,w))
    
    while True:
        
        frame=cvs.read()
        if frame is None:
            continue
        if camid==1:
            frame=cv2.flip(frame,1)
    
            
        print(frame.shape)
        img =cv2.resize(frame,(h,w)).astype(np.float32)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # framesamll=img
        
        img = img / 255
        
        print ('img',img.shape)
    
        tflite.set_g_index(0)
        tflite.setTensor_Fp32(img,h,w)
    
        tflite.invoke()
        
        pred = tflite.getTensor_Fp32(0)
        
        pred0=(pred[::2 ]).reshape(w,h)
        pred1=(pred[1::2]).reshape(w,h)
    
         
        back=((pred0)).copy()
        front=((pred1)).copy()
        
        mask=front-back
        mask[mask>0.0]=255
        mask[mask<=0.0]=0
        

        out1 = np.invert((back > 0.1) * 255)
        out2 = np.invert((front > 0.1) * 255)
        

        
        out1 = np.uint8(out1)
        out2 = np.uint8(out2)
        mask = np.uint8(mask)
        # out1 = cv2.resize(np.uint8(out1), (w, h))
        # out2 = cv2.resize(np.uint8(out2), (w, h))

        # out3 = cv2.ximgproc.jointBilateralFilter(out2, mask, 8, 150, 150)
        
        out3=mask.copy()
        
        out3=cv2.GaussianBlur(out3,(7,7),1)
        out3 = cv2.resize(out3,(tgt_size,tgt_size))
        
         
        out3=cv2.erode(out3, (3,3),iterations=4);
        
        kernel = np.ones((7,7),np.uint8)
        maskbig=cv2.dilate(out3, kernel,iterations=1);
        # kernel = np.ones((9,9),np.uint8)
        # masksmall=cv2.dilate(out3, kernel,iterations=1);
        # maskcircle = masksmall-out3
        maskcircle_back=maskbig-out3
        
        # out3==cv2.cvtColor(out3,cv2.COLOR_GRAY2BGR)
        out3 = out3/255
        # maskcircle=maskcircle/255
        maskcircle_back=maskcircle_back/255
        
        print ('out3',out3.shape)
        
        # out4=1.0-out3
        img1 =cv2.resize(frame,(hh,ww)).astype(np.float32)
        img1=img1/255
        in_tensor[:,:,0:3] = img1
        in_tensor[:,:,3] = masa
        
        tflite.set_g_index(1)
        tflite.setTensor_Fp32(in_tensor,input_shape1[1],input_shape1[1])
        tflite.invoke()
        
        pred = tflite.getTensor_Fp32(0)
        
        pred0=(pred[::2 ]).reshape(ww,hh)
        pred1=(pred[1::2]).reshape(ww,hh)
    
         
        back=pred0.copy()
        front=pred1.copy()        
        # back[back<0.0]=0
        # front[front<0.0]=0
        
        mask=front-back
        mask[mask>0.0]=255
        mask[mask<0.0]=0
        mask=cv2.erode(mask,(3,3),iterations=4)
        
        masa=mask/255        
        # mask=cv2.resize(mask,(,2*w))
        # out4=out3.copy()
        # out3[mask>0]=1
        # out3[mask<=0]=0
        # out3=masa
        
        # out4[out4>0]=255
        # out3[out3>0]=255
        
        # out4=np.uint8(out4)
        # out3=np.uint8(out3)
        
        
        
        # out3 = cv2.ximgproc.jointBilateralFilter(out4, out3, 8, 150, 150)
        # out3 = out3/255
        
        # out5=out4-out3
        # out5[out5<0]=0
        # out3=out3+out5
        
        # front[front<=0]=0
        # back[back<=0]=0
        
        
        # mask=front-back
        
        # mask[mask>0.0]=255
        # mask[mask<=0.0]=0    
        # print ('mask',mask.shape)
        # pred = tflite.getTensor_Fp32(0)
        
        # pred0=(pred[::2 ]).reshape(w,h)
        # pred1=(pred[1::2]).reshape(w,h)
    
        
        # back=((pred0)).copy().reshape(w,h)
        # front=((pred1)).copy().reshape(w,h)
        
        # out1 = np.invert((back > 0.5) * 255)
        # out2 = np.invert((front > 0.5) * 255)
        
        # out1 = cv2.resize(np.uint8(out1), (w, h))
        # out2 = cv2.resize(np.uint8(out2), (w, h))
        
        # out3 = cv2.ximgproc.jointBilateralFilter(out2, out1, 8, 75, 75)
        
        # front[front<=0]=0
        # back[back<=0]=0
        
        
        # mask=front
        # mask=(mask).astype('uint8')
        # mask=pred.copy().reshape(w,h)
        # mask = np.uint8(mask*255.0)
        # mask[mask>=0.0]=255
        # mask[mask<0.0]=0
        
        # mask=out3
        # ret,out3 = cv2.threshold(out3,100,255,cv2.THRESH_BINARY)
        # out3=cv2.GaussianBlur(out3,(3,3),1) 
        
        # mask=cv2.GaussianBlur(mask,(7,7),1) 
        
        # mask=out3
        frame=cv2.resize(frame,(tgt_size,tgt_size))
        out3=cv2.merge([out3,out3,out3])
        # maskcircle=cv2.merge([maskcircle,maskcircle,maskcircle])
        maskcircle_back=cv2.merge([maskcircle_back,maskcircle_back,maskcircle_back])
        
        print ('frame',frame.shape)
        print ('back',bgnd_mat.shape)
        
        # blur2=cv2.GaussianBlur(bgnd_mat, (25,25), 15)
       
        
        # framblur = frame*out3+(1-out3)*blur11
        
        # backblur = frame*out3+(1-out3)*bgnd_mat

        # blur1 = cv2.ximgproc.jointBilateralFilter(blur11.astype(np.float32),blur2.astype(np.float32),8,100,100)  
        # blur2 = cv2.ximgproc.jointBilateralFilter(framblur.astype(np.float32),bgnd_mat.astype(np.float32),4,180,180)
        # blur2=cv2.resize(blur2,(tgt_size,tgt_size))
        # blur1=cv2.resize(blur1,(tgt_size,tgt_size))
        
        
        
        
        # imgMultiply = frame*out3+(1-out3)*blur
        
        # # msk=cv2.resize(mask,(w,h))
        if mod==0:
            # blur1=cv2.GaussianBlur(bgnd_mat, (25,25), 15)
            # blur2=cv2.GaussianBlur(frame, (25,25), 15)
            # imgMultiply = frame*out3+(1-out3-maskcircle)*bgnd_mat+(maskcircle+maskcircle_back)*blur2
            # blur1=cv2.GaussianBlur(frame, (25,25), 15)
            
            # imgMultiply = frame*out3+(1-out3)*blur2
            
            imgMultiply = frame*out3+(1-out3)*bgnd_mat
            
            #imgMultiply = frame*out3+(1-out3-maskcircle-maskcircle_back)*bgnd_mat+maskcircle*(blur1)+maskcircle_back*(blur2)
            
            #imgMultiply = frame*out3+(1-out3-maskcircle-maskcircle_back)*bgnd_mat+maskcircle_back*(blur2+bgnd_mat)/2+maskcircle*blur1
            
            # imgMultiply=frame*out3+(1-out3-maskcircle-maskcircle_back)*bgnd_mat+(maskcircle*blur1+c*blur2)
            
            # imgMultiply = cv2.ximgproc.jointBilateralFilter(imgMultiply.astype(np.float32),imgMultiply1.astype(np.float32),8,5,5)
        #     dst=transfer_background_big(frame,bgnd_mat,mask)
        # elif mod==1:
        #     dst=seamlessclone(frame,bgnd_mat,mask,w,h)
        else :
            # blur1=cv2.GaussianBlur(frame, (25,25), 15)
            blur11=cv2.GaussianBlur(frame, (25,25), 15)  
            imgMultiply = frame*out3+(1-out3)*blur11
        #     
        
        # dst=transfer(frame, mask)
        # imgMultiply=imgMultiply.astype(np.float32)
        
        
        cvs.imshow(imgMultiply)
        sleep(1)    
        


if __name__ == '__main__':

    initcv(process)
    startcv(MyApp)
