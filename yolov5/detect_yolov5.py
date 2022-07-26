import os
import time
import numpy as np
import aidlite_gpu
import cv2
from aidlux_utils import detect_postprocess, draw_detect_res

det_model_path = "yolov5n_int8.dlc"
# 载入模型
aidlite = aidlite_gpu.aidlite()
# 载入检测模型
print(aidlite.ANNModel(det_model_path,
                       [1*320*320*3*4], 
                       [1*10*10*3*85*4], 
                       4, 
                       2))
        
nx     = 10
ny     = 10
stride = 32
na     = 3
no     = 85
anchors = np.array([[ 3.62500,  2.81250],
                    [ 4.87500,  6.18750],
                    [11.65625, 10.18750]], dtype=np.float32)
xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
grid = np.stack([xv, yv], 2)
grid = grid[np.newaxis,np.newaxis,...].repeat(na, axis=1) - 0.5
anchor_grid = (anchors * stride)[np.newaxis,:,np.newaxis,np.newaxis,:].repeat(ny, axis=2).repeat(nx, axis=3)                       
                      
im0 = cv2.imread('tiger_cat_320.jpg')                  
im = im0 / 255.0
im = im.astype(np.float32)[np.newaxis,...]   

aidlite.setInput_Float32(im,320,320)

while True:
    start=time.time()
    aidlite.invoke()
    invokeTime = time.time()-start
    print("invoke 时间:%f ms,invoke 帧率：%f fps"%(invokeTime*1000,1/invokeTime))
    res = aidlite.getOutput_Float32(0).reshape(1,10,10,3,85).transpose(0,3,1,2,4)
    y = 1 / (1 + np.exp(-res))
    y[..., 0:2] = (y[..., 0:2] * 2 + grid) * stride  # xy
    y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * anchor_grid
    y = y.reshape(1, -1, no)
    det_pred = detect_postprocess(y[0], 
                                  im0.shape, 
                                  [320, 320, 3], 
                                  conf_thres=0.5, 
                                  iou_thres=0.45)
    t = (time.time()-start)
    print("时间:%f ms帧率:%f fps"%(t*1000,1/t))
    res_img = draw_detect_res(im0, det_pred)
    cv2.imwrite("result.jpg", res_img)
                       