
from cvs import *

import numpy as np

import aidlite_gpu
aidlite=aidlite_gpu.aidlite()
back_img_path=('res/dock_vbig.jpeg','res/taj_vbig.jpg','res/sunset_vbig.jpg','res/test.jpg','res/bg1.jpg','res/bg2.jpg','res/bg3.jpg','res/bg4.jpg')

bgnd_mat=cv2.imread('res/dock_vbig.jpeg')
mod=0


class MyApp(App):
   
    def __init__(self, *args):
        super(MyApp, self).__init__(*args)
    
    def idle(self):
        self.aidcam0.update()
        
    def main(self):
        #creating a container VBox type, vertical (you can use also HBox or Widget)
        main_container = VBox(width=360, height=680, style={'margin':'0px auto'})
        main_container.css_width = "98%"
        self.aidcam0 = OpencvVideoWidget(self, width=350, height=400)
        self.aidcam0.css_width = "98%"
        self.aidcam0.style['margin'] = '10px'
        
        i=0
        exec("self.aidcam%(i)s = OpencvVideoWidget(self)" % {'i': i})
        exec("self.aidcam%(i)s.identifier = 'aidcam%(i)s'" % {'i': i})
        eval("main_container.append(self.aidcam%(i)s)" % {'i': i})
        
        main_container.append(self.aidcam0)
        
        self.lbl = Label('点击图片选择你喜欢的虚拟背景：')
        main_container.append(self.lbl)
        
        m_container = HBox(width=360, height=130, style={'margin':'0px auto'})
        m_container.css_width = "98%"
        self.img11 = Image('/res:'+os.getcwd()+'/'+back_img_path[4], width=80,height=80, margin='10px')
        self.img11.onclick.do(self.on_img11_clicked)
        m_container.append(self.img11)
        
        self.img12 = Image('/res:'+os.getcwd()+'/'+back_img_path[5], width=80,height=80, margin='10px')
        self.img12.onclick.do(self.on_img12_clicked)
        m_container.append(self.img12)
        
        self.img13 = Image('/res:'+os.getcwd()+'/'+back_img_path[6],width=80, height=80, margin='10px')
        self.img13.onclick.do(self.on_img13_clicked)
        m_container.append(self.img13)

        self.img14 = Image('/res:'+os.getcwd()+'/'+back_img_path[7], width=80,height=80,margin='10px')
        self.img14.onclick.do(self.on_img14_clicked)
        m_container.append(self.img14)
        
        bottom_container = HBox(width=360, height=130, style={'margin':'0px auto'})
        bottom_container.css_width = "98%"
        self.img1 = Image('/res:'+os.getcwd()+'/'+back_img_path[0],  width=80,height=80, margin='10px')
        self.img1.onclick.do(self.on_img1_clicked)
        bottom_container.append(self.img1)
        
        self.img2 = Image('/res:'+os.getcwd()+'/'+back_img_path[1],  width=80,height=80, margin='10px')
        self.img2.onclick.do(self.on_img2_clicked)
        bottom_container.append(self.img2)
        
        self.img3 = Image('/res:'+os.getcwd()+'/'+back_img_path[2],  width=80,height=80, margin='10px')
        self.img3.onclick.do(self.on_img3_clicked)
        bottom_container.append(self.img3)

        self.img4 = Image('/res:'+os.getcwd()+'/'+back_img_path[3], height=80, width=80,margin='10px')
        self.img4.onclick.do(self.on_img4_clicked)
        bottom_container.append(self.img4)
        
        
        b_container = HBox(width=360, height=100, style={'margin':'0px auto'})
        b_container.css_width = "98%"
        self.bt1 = Button('抠图穿越', width=100, height=30, margin='10px')
        self.bt1.onclick.do(self.on_button_pressed1)

        self.bt3 = Button('背景虚化', width=100, height=30, margin='10px')
        self.bt3.onclick.do(self.on_button_pressed3) 
        
        main_container.append(m_container)
        main_container.append(bottom_container)
        b_container.append(self.bt1)
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

    def on_img11_clicked(self, widget):
        global bgnd_mat
        bgnd=cv2.imread(back_img_path[4])
        bgnd_mat=cv2.resize(bgnd,(512, 512))
        
    def on_img12_clicked(self, widget):
        global bgnd_mat
        bgnd=cv2.imread(back_img_path[5])
        bgnd_mat=cv2.resize(bgnd,(512, 512))
        
    def on_img13_clicked(self, widget):
        global bgnd_mat
        bgnd=cv2.imread(back_img_path[6])       
        bgnd_mat=cv2.resize(bgnd,(512, 512))
        
    def on_img14_clicked(self, widget):
        global bgnd_mat
        bgnd=cv2.imread(back_img_path[7])       
        bgnd_mat=cv2.resize(bgnd,(512, 512))   
        
    def on_button_pressed1(self, widget):
        global mod
        mod=0
        
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
    model_path="res/portrait_segmentation.tflite"
    print('gpu0:',aidlite.ANNModel(model_path,inShape,outShape,4,0))

    camid=1
    cap=cvs.VideoCapture(camid)
    
    tgt_size=512
    global bgnd_mat
    bgnd_mat=cv2.resize(bgnd_mat,(tgt_size, tgt_size))
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
        frame=img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        img = img / 255

        aidlite.setTensor_Fp32(img,input_shape[1],input_shape[1])
    
        aidlite.invoke()
    
        pred = aidlite.getTensor_Fp32(0)
        
        pred0=(pred[::2 ]).reshape(w,h)
        pred1=(pred[1::2]).reshape(w,h)
    
        back=((pred0))
        front=((pred1))
        
        mask=front-back
        mask[mask>0.0]=255
        mask[mask<=0.0]=0        

        # out1 = np.invert((back > 0.5) * 255)
        out2 = np.invert((front > 0.5) * 255)
        # out1 = np.uint8(out1)
        out2 = np.uint8(out2)
        mask=np.uint8(mask)
        
        out2=cv2.resize(out2,(256,256))
        mask = cv2.resize(mask,(256,256))
        out3 = cv2.ximgproc.jointBilateralFilter(out2, mask, 8, 100, 100)       
        out3 = cv2.resize(out3,(512,512))
        
        # out3=mask.copy()
        
        out3=cv2.GaussianBlur(out3,(7,7),1) 
        out3 = out3/255
        masksmall=cv2.erode(mask, (3,3),iterations=1);
           
        out3=cv2.merge([out3,out3,out3])
        # out3 = (cv2.cvtColor(out3, cv2.COLOR_GRAY2RGB))
        # out3 = out3/255
        # dst = frame*out3+(1-out3)*bgnd_mat
        
        # frame = cv2.resize(frame,(256,256))
        # bgnd_mat = cv2.resize(bgnd_mat,(256,256))
        
        if mod==0:
            dst = frame*out3+(1-out3)*bgnd_mat
        else :
            blur= cv2.GaussianBlur(frame, (27,27), 15) 
            dst = frame*out3+(1-out3)*blur

        cvs.imshow(dst)
        sleep(1)    
        


if __name__ == '__main__':

    initcv(startcv, MyApp)
    process()