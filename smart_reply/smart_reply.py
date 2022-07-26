import execute
import jieba
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
        self.txt = gui.TextInput(width=260, height=22)
        self.txt.set_text('TEXTAREA')
        self.txt.onchange.do(self.on_text_area_change)
        
        self.bt = gui.Button('Send', width=80, height=22)
        self.bt.onclick.do(self.on_button_pressed)
        horizontalContainer.append([self.txt, self.bt])
        
        # self.lbl = gui.Label('LABEL!', width=404, height=50, margin='10px')
        
        self.btClose = gui.Button('Exit', width=340, height=22, margin='10px')
        self.btClose.onclick.do(self.server_close)
        verticalContainer.append([self.log, horizontalContainer, self.btClose])
        
        return verticalContainer
        
    def on_button_pressed(self, widget):
        self.info += self.req_msg + '\n\r'
        self.log.set_text(self.info)
        req_msg=" ".join(jieba.cut(self.req_msg))
        
        res_msg = execute.predict(req_msg)
        #将unk值的词用微笑符号袋贴
        res_msg = res_msg.replace('_UNK', '^_^')
        res_msg=res_msg.strip()
        
        # 如果接受到的内容为空，则给出相应的回复
        if res_msg == ' ':
          res_msg = '请与我聊聊天吧'
        
        # print('question message : ', self.req_msg)
        # print('responce message : ', res_msg)
        self.info += res_msg + '\n\r' 
        
        # self.lbl.set_text(res_msg)
        self.log.set_text(self.info)
        
        
    def on_text_area_change(self, widget, newValue):
        self.req_msg = newValue
        
    def server_close(self, widget):
        pname = 'smart_reply.py'
        urlopen('http://127.0.0.1:9000/?name=kill~' + pname)
        
if __name__ == "__main__":
    droid.showCustomDialog(5556)
    # sio.emit('request_open_window', {
    #             'data': {'url': 'http://0.0.0.0:' + str(5556), 'title': 'aaa', 'icon': '', 'camid': 0}})
    start(MyApp, update_interval=0.1, address='0.0.0.0', port=5556, start_browser=False, enable_file_cache=False)

