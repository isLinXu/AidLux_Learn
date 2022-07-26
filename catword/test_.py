#coding=utf-8
'''
Author: your name
Date: 2021-10-14 18:34:51
LastEditTime: 2021-10-15 16:02:56
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath:
'''
import predict
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

class MyApp(App):
    cnn_model = predict.CnnModel()
    def __init__(self, *args):
        super(MyApp, self).__init__(*args)
        self.cnn_model = predict.CnnModel()

    def main(self):
        self.req_msg = ''
        self.info = ''
        verticalContainer = gui.Container(width=360, height=600, style={'display': 'block', 'overflow': 'hidden','text-align': 'center'})
        self.log = gui.Label('', width=360, height=360, margin='10px', style={'white-space':'pre'})
        
        horizontalContainer = gui.Container(width=360, height=36, layout_orientation=gui.Container.LAYOUT_HORIZONTAL, margin='10px', style={'display': 'block', 'overflow': 'auto'})        
        # 输入框
        self.txt = gui.TextInput(width=260, height=22)
        self.txt.set_text('待分类的文本') #
        self.txt.onchange.do(self.on_text_area_change)
        
        # 答案放置处
        # 考虑到答案可能较长的应用（古文生成等），此标签属性可能不适用，比如答案可能会漫出容器。请根据需要调整width和height两个属性即可
        self.lbl = gui.Label(" ",width=360, height=100, margin='10px')
        
        # Log按钮
        self.btLog = gui.Button('Log', width=40, height=22)
        self.btLog.onclick.do(self.show_log)
        
        # 发送按钮
        self.bt = gui.Button('确认', width=40, height=22)
        self.bt.onclick.do(self.on_button_pressed)
        
        
        
        # 装入水平容器，形成一行
        horizontalContainer.append([self.txt, self.btLog, self.bt])
        
        self.btClose = gui.Button('退出', width=340, height=22, margin='10px')
        self.btClose.onclick.do(self.server_close)
        verticalContainer.append([self.log, horizontalContainer, self.lbl, self.btClose])
        
        return verticalContainer
        
    def on_button_pressed(self, widget):
        req_msg = self.req_msg
        # cnn_model = predict.CnnModel()
        res_msg = "类别："+self.cnn_model.predict(req_msg) #
        # 如果接受到的内容为空，则给出相应的回复
        if res_msg == '输入待分类的文本':
          self.txt.set_text = '请清除此文本框内容，输入待分类的文本' #
        
        # 结果记录，对用户透明
        self.info += req_msg + '\n\r'+ res_msg + '\n\r' #
        # self.log.set_text(self.info)
        self.lbl.set_text(res_msg)
        
    def on_text_area_change(self, widget, newValue):
        self.req_msg = newValue
    
    def show_log(self, widget):
        self.log.set_text(self.info)
        
    def server_close(self, widget):
        pname = 'test_.py' # 
        urlopen('http://127.0.0.1:9000/?name=kill~' + pname)
        
    
        
if __name__ == "__main__":
    droid.showCustomDialog(5558)
    # sio.emit('request_open_window', {
    #             'data': {'url': 'http://0.0.0.0:' + str(5558), 'title': 'aaa', 'icon': '', 'camid': 0}})
    start(MyApp, update_interval=0.1, address='0.0.0.0', port=5558, start_browser=False, enable_file_cache=False)
    

# test_demo = ['三星ST550以全新的拍摄方式超越了以往任何一款数码相机',
#                  '热火vs骑士前瞻：皇帝回乡二番战 东部次席唾手可得新浪体育讯北京时间3月30日7:00']




