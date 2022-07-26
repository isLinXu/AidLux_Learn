#coding=utf-8
'''
Author: Tommy
Date: 2021-10-14 18:34:51
LastEditTime: 2021-10-15 16:02:56
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath:
'''
from jiayan import load_lm
from jiayan import CRFPunctuator
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
    def __init__(self, *args):
        super(MyApp, self).__init__(*args)

    def main(self):
        self.req_msg = ''
        self.info = ''
        verticalContainer = gui.Container(width=360, height=600, style={'display': 'block', 'overflow': 'hidden','text-align': 'center'})
        self.log = gui.Label('', width=360, height=360, margin='10px', style={'white-space':'pre'})
        
        horizontalContainer = gui.Container(width=360, height=26, layout_orientation=gui.Container.LAYOUT_HORIZONTAL, margin='10px', style={'display': 'block', 'overflow': 'auto'})        
        # 输入框
        self.txt = gui.TextInput(width=260, height=22)
        self.txt.set_text('需要断句的古汉语文本')
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
        lm = load_lm('jiayan_models/jiayan.klm')
        punctuator = CRFPunctuator(lm, 'jiayan_models/cut_model')
        punctuator.load('jiayan_models/punc_model')
        res_msg = punctuator.punctuate(req_msg)
        # 如果接受到的内容为空，则给出相应的回复
        if res_msg == ' ':
          res_msg = '请清除此文本框，输入需要断句的古汉语文本'
        
        # 结果记录，对用户透明
        self.info += res_msg + '\n\r'
        # self.log.set_text(self.info)
        self.lbl.set_text(res_msg)
        
    def on_text_area_change(self, widget, newValue):
        self.req_msg = newValue
    
    def show_log(self, widget):
        pname = 'jiayan_test_.py'
        self.log.set_text(self.info)
        
    def server_close(self, widget):
        pname = 'jiayan_test_.py'
        urlopen('http://127.0.0.1:9000/?name=kill~' + pname)
        
    
        
if __name__ == "__main__":
    droid.showCustomDialog(5558)
    # sio.emit('request_open_window', {
    #             'data': {'url': 'http://0.0.0.0:' + str(5558), 'title': 'aaa', 'icon': '', 'camid': 0}})
    start(MyApp, update_interval=0.1, address='0.0.0.0', port=5558, start_browser=False, enable_file_cache=False)


# test_demo = '天下大乱贤圣不明道德不一天下多得一察焉以自好譬如耳目皆有所明不能相通犹百家众技也皆有所长时有所用虽然不该不遍一之士也判天地之美析万物之理察古人之全寡能备于天地之美称神之容是故内圣外王之道暗而不明郁而不发天下之人各为其所欲焉以自为方悲夫百家往而不反必不合矣后世之学者不幸不见天地之纯古之大体道术将为天下裂'
