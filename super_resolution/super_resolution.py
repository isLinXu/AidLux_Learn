import cv2
import android
droid = android.Android()
import numpy as np
import PIL.Image as Image
import base64
import tflite_gpu
tflite=tflite_gpu.tflite()

import remi.gui as gui
from urllib.request import urlopen
from remi import start, App
from threading import Timer
try:
    import socketio

    sio = socketio.Client()
    sio.connect('ws://0.0.0.0:9909')
    socket_connection = True
except:
    socket_connection = False
    
def base64_to_style_image(base64_image):
    return 'data:image/png;base64,'+base64_image
    
class MyApp(App):
    def __init__(self, *args):
        super(MyApp, self).__init__(*args)

    def main(self):
        inShape =[1*50*50*3*4,]
        outShape= [1*200*200*3*4,]
        model_path="ESRGAN.tflite"
        print('gpu:',tflite.NNModel(model_path,inShape,outShape,4,0))

        verticalContainer = gui.Container(width=404, height=700, style={'display': 'block', 'overflow': 'hidden','text-align': 'center'})
        
        self.img_lr = gui.Image('', width=50,height=50,margin='1px')
        verticalContainer.append(self.img_lr)
        
        self.btFileDiag = gui.Button('Select Image', width='100%', height=30, margin='0px')
        self.btFileDiag.onclick.do(self.open_fileselection_dialog)
        verticalContainer.append(self.btFileDiag)
        
        horizontalContainer = gui.Container(width=410, height=205, layout_orientation=gui.Container.LAYOUT_HORIZONTAL, margin='0px', style={'display': 'block', 'overflow': 'auto'})        
        self.img_hr = gui.Image('',width=200, height=200, margin='1px')
        self.img_x2 = gui.Image('',width=200, height=200, margin='1px')
        horizontalContainer.append(self.img_x2)
        horizontalContainer.append(self.img_hr)
        
        verticalContainer.append(horizontalContainer)
        
        
        self.btClose = gui.Button('Exit', width=404, height=30, margin='0px')
        self.btClose.onclick.do(self.server_close)
        verticalContainer.append(self.btClose)
        
        return verticalContainer

    def open_fileselection_dialog(self, widget):
        print('open_fileselection_dialog')
        self.fileselectionDialog = gui.FileSelectionDialog('File Selection Dialog', 'Select files and folders', False,
                                                          '.')
        self.fileselectionDialog.confirm_value.do(
            self.on_fileselection_dialog_confirm)

        # here is returned the Input Dialog widget, and it will be shown
        self.fileselectionDialog.show(self)
                
    def on_fileselection_dialog_confirm(self, widget, filelist):
        print('filelist', filelist)
        if (len(filelist) > 0):
            file_name = filelist[0]
            with open(filelist[0], "rb") as f:
                lr, x2, hr = self.imageDemo(file_name)
                print(lr.shape, x2.shape, hr.shape)
                lr = cv2.imencode('.jpg',lr)[1]
                x2 = cv2.imencode('.jpg',x2)[1]
                hr = cv2.imencode('.jpg',hr)[1]
                bs64_str_lr = str(base64.b64encode(lr))[2:-1]
                bs64_str_x2 = str(base64.b64encode(x2))[2:-1]
                bs64_str_hr = str(base64.b64encode(hr))[2:-1]
                if not isinstance(bs64_str_lr, str):
                    bs64_str_lr = bs64_str_lr.decode()
                if not isinstance(bs64_str_x2, str):
                    bs64_str_x2 = bs64_str_x2.decode()
                if not isinstance(bs64_str_hr, str):
                    bs64_str_hr = bs64_str_hr.decode()
                self.img_lr.set_image(base64_to_style_image(bs64_str_lr))
                self.img_x2.set_image(base64_to_style_image(bs64_str_x2))
                self.img_hr.set_image(base64_to_style_image(bs64_str_hr))

        
    def imageDemo(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (50, 50))
        img_x2 = cv2.resize(img, (200, 200))
        img_np = np.expand_dims(img.astype(np.float32), 0)
        tflite.setTensor_Fp32(img_np,50,50)
        tflite.invoke()
        pred = tflite.getTensor_Fp32(0).reshape(200,200,3)
        pred = np.clip(pred, 0, 255)
        pred = np.round(pred)
        pred = pred.astype(np.uint8)
        return img, img_x2, pred
    #     cv2.imwrite('res.jpg', pred)
    
    def server_close(self, widget):
        pname = 'super_resolution.py'
        urlopen('http://127.0.0.1:9000/?name=kill~' + pname)
        
if __name__ == "__main__":
    droid.showCustomDialog(5556)
    # sio.emit('request_open_window', {
    #             'data': {'url': 'http://0.0.0.0:' + str(5556), 'title': 'aaa', 'icon': '', 'camid': 0}})
    start(MyApp, update_interval=0.1, address='0.0.0.0', port=5556, start_browser=False, enable_file_cache=False)

