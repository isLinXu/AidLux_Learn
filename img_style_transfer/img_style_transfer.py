import cv2
import numpy as np
import PIL.Image as Image
import base64
import tensorflow as tf
import matplotlib.pyplot as plt
import android
droid = android.Android()

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

        verticalContainer = gui.Container(width=380, height=520, style={'display': 'block', 'overflow': 'hidden','text-align': 'center'})
        
        horizontalContainer = gui.Container(width=380, height=190, layout_orientation=gui.Container.LAYOUT_HORIZONTAL, margin='0px', style={'display': 'block', 'overflow': 'auto'}) 
        self.img_lr = gui.Image('', width=180,height=180,margin='1px')
        self.img_x2 = gui.Image('',width=180, height=180, margin='1px')
        horizontalContainer.append(self.img_lr)
        horizontalContainer.append(self.img_x2)
        verticalContainer.append(horizontalContainer)
        
        self.btFileDiag = gui.Button('Change Image', width='100%', height=30, margin='0px')
        self.btFileDiag.onclick.do(self.open_fileselection_dialog)
        verticalContainer.append(self.btFileDiag)
       
        self.img_res = gui.Image('',width=256, height=256, margin='1px')
        verticalContainer.append(self.img_res)
        
        self.btClose = gui.Button('Exit', width='100%', height=30, margin='0px')
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
                self.img_res.set_image(base64_to_style_image(bs64_str_hr))

    def on_fileselection_dialog_confirm2(self, widget, filelist):
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
                self.img_res.set_image(base64_to_style_image(bs64_str_hr))
    


    def imageDemo(self, content_path):
        
        style_path = './style_img/style19.jpg'  # 选择风格路径
        
        img = cv2.imread(content_path)
        
        style = cv2.imread(style_path)

        style_predict_path = 'style_prediction_256_fp16.tflite'
        style_transform_path = 'style_transfer_256_fp16.tflite'
        
        def load_img(path_to_img):
            img = tf.io.read_file(path_to_img)
            img = tf.io.decode_image(img, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = img[tf.newaxis, :]
            return img
    
        def preprocess_image(image, target_dim):
            # Resize the image so that the shorter dimension becomes 256px.
            shape = tf.cast(tf.shape(image)[1:-1], tf.float32)
            short_dim = min(shape)
            scale = target_dim / short_dim
            new_shape = tf.cast(shape * scale, tf.int32)
            image = tf.image.resize(image, new_shape)
    
            # Central crop the image.
            image = tf.image.resize_with_crop_or_pad(image, target_dim, target_dim)
    
            return image
    
        def run_style_predict(preprocessed_style_image, style_predict_path):
            # Load the model.
            interpreter = tf.lite.Interpreter(model_path=style_predict_path)
    
            # Set model input.
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            interpreter.set_tensor(input_details[0]["index"], preprocessed_style_image)
    
            # Calculate style bottleneck.
            interpreter.invoke()
            style_bottleneck = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
            return style_bottleneck
    
        def run_style_transform(style_bottleneck, preprocessed_content_image, style_transform_path):
            # Load the model.
            interpreter = tf.lite.Interpreter(model_path=style_transform_path)
    
            # Set model input.
            input_details = interpreter.get_input_details()
            interpreter.allocate_tensors()
    
            # Set model inputs.
            interpreter.set_tensor(input_details[0]["index"], preprocessed_content_image)
            interpreter.set_tensor(input_details[1]["index"], style_bottleneck)
            interpreter.invoke()
    
            # Transform content image.
            stylized_image = interpreter.tensor(interpreter.get_output_details()[0]["index"])()
            return stylized_image
        
        
        # Load the input images.
        content_image = load_img(content_path)
        style_image = load_img(style_path)

        # Preprocess the input images.
        preprocessed_content_image = preprocess_image(content_image, 384)
        preprocessed_style_image = preprocess_image(style_image, 256)

        print('Style Image Shape:', preprocessed_style_image.shape)
        print('Content Image Shape:', preprocessed_content_image.shape)

        # Calculate style bottleneck for the preprocessed style image.
        style_bottleneck = run_style_predict(preprocessed_style_image, style_predict_path)
        print('Style Bottleneck Shape:', style_bottleneck.shape)

        # Stylize the content image using the style bottleneck.
        stylized_image = run_style_transform(style_bottleneck, preprocessed_content_image, style_transform_path)  # (1, 384, 384, 3) <class 'numpy.ndarray'> range:[0, 1]
        stylized_image = np.squeeze(stylized_image, axis=0)
        stylized_image = stylized_image * 255
        # pred = np.transpose(pred, (1, 2, 0))
        stylized_image = stylized_image.astype(np.uint8)
        
        return img, style, stylized_image
    
    
    def server_close(self, widget):
        pname = 'img_style_transfer.py'
        urlopen('http://127.0.0.1:9000/?name=kill~' + pname)
        
if __name__ == "__main__":
    # sio.emit('request_open_window', {
    #             'data': {'url': 'http://0.0.0.0:' + str(5556), 'title': 'aaa', 'icon': '', 'camid': 0}})
    droid.showCustomDialog(5556)
    start(MyApp, update_interval=0.1, address='0.0.0.0', port=5556, start_browser=False, enable_file_cache=False)

