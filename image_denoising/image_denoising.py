import PIL.Image as Image
import base64
import android
droid = android.Android()

import remi.gui as gui
from urllib.request import urlopen
from remi import start, App
from threading import Timer

import matplotlib.pyplot as plt
import cv2
import torchvision
import numpy as np
import torch

from UNet import UNet


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
        
        verticalContainer = gui.Container(width=404, height=700, style={'display': 'block', 'overflow': 'hidden','text-align': 'center'})
        
        self.img_lr = gui.Image('', width=256,height=256,margin='1px')
        verticalContainer.append(self.img_lr)
        
        self.btFileDiag = gui.Button('Select Image', width='100%', height=30, margin='0px')
        self.btFileDiag.onclick.do(self.open_fileselection_dialog)
        verticalContainer.append(self.btFileDiag)
        
        self.img_hr = gui.Image('',width=256, height=256, margin='1px')
        verticalContainer.append(self.img_hr)
        
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
                lr, hr = self.imageDemo(file_name)
                print(lr.shape, hr.shape)
                lr = cv2.imencode('.jpg',lr)[1]
                
                hr = cv2.imencode('.jpg',hr)[1]
                bs64_str_lr = str(base64.b64encode(lr))[2:-1]
                
                bs64_str_hr = str(base64.b64encode(hr))[2:-1]
                if not isinstance(bs64_str_lr, str):
                    bs64_str_lr = bs64_str_lr.decode()
                
                if not isinstance(bs64_str_hr, str):
                    bs64_str_hr = bs64_str_hr.decode()
                self.img_lr.set_image(base64_to_style_image(bs64_str_lr))
                self.img_hr.set_image(base64_to_style_image(bs64_str_hr))

        
    def imageDemo(self, image_path):
        # 使用opencv读取图像
        
        img = cv2.imread(image_path)     # np.array, (H x W x C), [0, 255], BGR
        # img = cv2.resize(img, (256, 256))
        
        checkpoint = torch.load('./checkpoints/chk_2_1_0.5.pt', map_location=torch.device('cpu'))

        model_test = UNet(in_channels=3, out_channels=3).double()
        model_test.load_state_dict(checkpoint['model_state_dict'])
        model_test = model_test.cpu()
        
        
        torch.set_default_tensor_type(torch.DoubleTensor)
        print(img.shape)
        tensor_cv = torch.from_numpy(np.transpose(img, (2, 0, 1)))   # (C x H x W)
        tensor_cv = tensor_cv.unsqueeze(0)
        tensor_cv = tensor_cv / 127.5 - 1   # range: [-1, 1]
        
        # print(tensor_cv.shape, tensor_cv)
        denoise_img = torchvision.utils.make_grid(model_test(tensor_cv))
        print('----denoise shape:', denoise_img.shape, type(denoise_img))
        
        pred = (denoise_img + 1) * 127.5     # unnormalize [0, 255]
        pred = pred.detach().numpy()
        pred = np.transpose(pred, (1, 2, 0))
        pred = pred.astype(np.uint8)
        print('----pred shape:', pred.shape)
        # cv2.imwrite('res_test.jpg', pred)
        return img, pred
        
    
    def server_close(self, widget):
        pname = 'image_denoising.py'
        urlopen('http://127.0.0.1:9000/?name=kill~' + pname)
        
if __name__ == "__main__":
    droid.showCustomDialog(5556)
    # sio.emit('request_open_window', {
    #             'data': {'url': 'http://0.0.0.0:' + str(5556), 'title': 'Deniose', 'icon': '', 'camid': 0}})
    start(MyApp, update_interval=0.1, address='0.0.0.0', port=5556, start_browser=False, enable_file_cache=False)

