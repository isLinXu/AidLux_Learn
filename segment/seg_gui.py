
from cvs import *

import numpy as np

import tflite_gpu
tflite=tflite_gpu.tflite()
back_img_path=('res/dock_vbig.jpeg','res/taj_vbig.jpg','res/sunset_vbig.jpg','/home/cvs/test.jpg')

bgnd_mat=cv2.imread('res/dock_vbig.jpeg')
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
    # cv::dilate(mask_mat, mask_mat, cv::Mat(), cv::Point(-1, -1), 2, 1, 1);
    
    mask_mat = cv2.resize(mask, (w, h),interpolation=cv2.INTER_NEAREST)
    input_mat = cv2.resize(input, (w,h),interpolation=cv2.INTER_NEAREST)

    roi=find_max_region(mask_mat)
    mask_mat=cv2.cvtColor(mask_mat,cv2.COLOR_GRAY2BGR)

    input_mat=input_mat[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]] 
    mask_mat=mask_mat[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]]      
    mask_mat[mask_mat>0]=255
    
    width, height, channels = bgnd_mat.shape
    center =( width-input_mat.shape[1]//2, height- input_mat.shape[0]// 2)


    normal_clone = cv2.seamlessClone(input_mat, bgnd_mat, mask_mat, center, cv2.NORMAL_CLONE)#MIXED_CLONE NORMAL_CLONE
    
    return normal_clone
    

def transfer_background_big(input,bgd, mask):

    mask = np.uint8(mask*255.0)
    
    _,mask=cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    
    mask=cv2.dilate(mask, None,iterations=1);
    
 
    kernel = np.ones((9,9),np.uint8)
    masksmall=cv2.erode(mask, kernel,iterations=2);
    # masksmall=mask
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



    alpha = 0.5
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
        
        bottom_container = HBox(width=360, height=130, style={'margin':'0px auto'})
        self.img1 = Image('/res:'+os.getcwd()+'/'+back_img_path[0], height=70, margin='10px')
        self.img1.onclick.do(self.on_img1_clicked)
        bottom_container.append(self.img1)
        
        self.img2 = Image('/res:'+os.getcwd()+'/'+back_img_path[1], height=70, margin='10px')
        self.img2.onclick.do(self.on_img2_clicked)
        bottom_container.append(self.img2)
        
        self.img3 = Image('/res:'+os.getcwd()+'/'+back_img_path[2], height=70, margin='10px')
        self.img3.onclick.do(self.on_img3_clicked)
        bottom_container.append(self.img3)

        self.img4 = Image('/res:'+back_img_path[3], height=70, width=80,margin='10px')
        self.img4.onclick.do(self.on_img4_clicked)
        bottom_container.append(self.img4)
        
        b_container = HBox(width=360, height=100, style={'margin':'0px auto'})
        
        self.bt1 = Button('抠图模式', width=70, height=30, margin='10px')
        self.bt1.onclick.do(self.on_button_pressed1)
        
        self.bt2 = Button('渲染模式', width=70, height=30, margin='10px')
        self.bt2.onclick.do(self.on_button_pressed2)        

        self.bt3 = Button('着色模式', width=70, height=30, margin='10px')
        self.bt3.onclick.do(self.on_button_pressed3) 
        
        main_container.append(bottom_container)
        b_container.append(self.bt1)
        b_container.append(self.bt2)
        b_container.append(self.bt3)
        main_container.append(b_container)

        return main_container
        
    def on_img1_clicked(self, widget):
        global bgnd_mat
        bgnd=cv2.imread(back_img_path[0])
        bgnd_mat=cv2.resize(bgnd,(512, 512))
        
    def on_img2_clicked(self, widget):
        global bgnd_mat
        bgnd=cv2.imread(back_img_path[1])
        bgnd_mat=cv2.resize(bgnd,(512, 512))
        
    def on_img3_clicked(self, widget):
        global bgnd_mat
        bgnd=cv2.imread(back_img_path[2])       
        bgnd_mat=cv2.resize(bgnd,(512, 512))
        
    def on_img4_clicked(self, widget):
        global bgnd_mat
        bgnd=cv2.imread(back_img_path[3])       
        bgnd_mat=cv2.resize(bgnd,(512, 512))   
        
    def on_button_pressed1(self, widget):
        global mod
        mod=0
        
    def on_button_pressed2(self, widget):
        global mod
        mod=1   
        
    def on_button_pressed3(self, widget):
        global mod
        mod=2
        

def process():
    cvs.setCustomUI()

    w=512
    h=512
    input_shape=[w,h]
    inShape =[1 * w * h *3*4,]
    outShape= [1 * w*h*2*4,]
    model_path="portrait_segmentation.tflite"
    print('gpu:',tflite.NNModel(model_path,inShape,outShape,4,0))
    
    camid=1
    cap=cvs.VideoCapture(camid)
    
    tgt_size=512
    global bgnd_mat
    bgnd_mat=cv2.resize(bgnd_mat,(tgt_size, tgt_size))
    # bgd = cv2.cvtColor(bgnd_mat, cv2.COLOR_BGR2RGB)/255.0
    fcounts=0
    fframes=[None,None,None,None]
    fmax=[0,0,0,0]
    
    while True:
        
        frame=cvs.read()
        if frame is None:
            continue
        if camid==1:
            frame=cv2.flip(frame,1)

            
        img =cv2.resize(frame,(input_shape[0],input_shape[1]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        # img=cv2.GaussianBlur(img,(7,7),1)
        img = img / 255
        
        print ('img',img.shape)
        
    
        # interpreter.set_tensor(input_details[0]['index'], img[np.newaxis,:,:,:])
        tflite.setTensor_Fp32(img,input_shape[1],input_shape[1])
    
        tflite.invoke()
    
        pred = tflite.getTensor_Fp32(0)
        
        pred0=(pred[::2 ]).reshape(w,h)
        pred1=(pred[1::2]).reshape(w,h)
    
         
        back=((pred0)).copy()
        front=((pred1)).copy()
        
        # out1 = np.invert((back > 0.5) * 255)
        # out2 = np.invert((front > 0.5) * 255)
        # out1 = np.uint8(out1)
        # out2 = np.uint8(out2)
        # # out1 = cv2.resize(np.uint8(out1), (w, h))
        # # out2 = cv2.resize(np.uint8(out2), (w, h))

        # out3 = cv2.ximgproc.jointBilateralFilter(out2, out1, 8, 75, 75)        
        
        
        front[front<=0]=0
        back[back<=0]=0
        
        
        mask=front-back
        # mask[front>0.7]=255
        mask[mask>0.0]=255
        
        mask[mask<=0.0]=0
        
        # out3[out3>200]=255
        # mask=out3
        mask=cv2.GaussianBlur(mask,(7,7),1)
        
        frame=cv2.resize(frame,(tgt_size,tgt_size))
        if mod==0:
            dst=transfer_background_big(frame,bgnd_mat,mask)
        elif mod==1:
            dst=seamlessclone(frame,bgnd_mat,mask,360,360)
        else :
            dst=transfer(frame,mask)
        
        # dst=harmonize(dst,mask)
        cvs.imshow(dst)
        sleep(1)    
        


if __name__ == '__main__':

    initcv(process)
    startcv(MyApp)
