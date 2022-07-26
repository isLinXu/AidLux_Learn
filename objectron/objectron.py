from cvs import *
import time
import sys
import numpy as np
from scipy.ndimage.filters import maximum_filter
import aidlite_gpu

aidlite = aidlite_gpu.aidlite()

def detect_peak(image, filter_size=5, order=0.5):
    local_max = maximum_filter(image, footprint=np.ones((filter_size, filter_size)), mode='constant')
    detected_peaks = np.ma.array(image,mask=~(image == local_max))
    
    temp = np.ma.array(detected_peaks, mask=~(detected_peaks >= detected_peaks.max() * order))
    peaks_index = np.where((temp.mask != True))
    return peaks_index

def decode(hm, displacements, threshold=0.8):
    hm = hm.reshape(40,30)
    displacements = displacements.reshape(1, 40, 30, 16)
    peaks = detect_peak(hm)
    peakX = peaks[1]
    peakY = peaks[0]

    scaleX = hm.shape[1]
    scaleY = hm.shape[0]
    objs = []
    for x,y in zip(peakX, peakY):
        conf = hm[y,x]
        if conf<threshold:
            continue
        points=[]
        for i in range(8):
            dx = displacements[0, y, x, i*2]
            dy = displacements[0, y, x, i*2+1]
            points.append((x/scaleX+dx, y/scaleY+dy))
        objs.append(points)
    return objs
    
def draw_box(image, pts):
    scaleX = image.shape[1]
    scaleY = image.shape[0]

    lines = [(0,1), (1,3), (0,2), (3,2), (1,5), (0,4), (2,6), (3,7), (5,7), (6,7), (6,4), (4,5)]
    for line in lines:
        pt0 = pts[line[0]]
        pt1 = pts[line[1]]
        pt0 = (int(pt0[0]*scaleX), int(pt0[1]*scaleY))
        pt1 = (int(pt1[0]*scaleX), int(pt1[1]*scaleY))
        cv2.line(image, pt0, pt1, (255 ,245, 0))
    
    for i in range(8):
        pt = pts[i]
        pt = (int(pt[0]*scaleX), int(pt[1]*scaleY))
        cv2.circle(image, pt, 1,(255 ,245, 0), -1)
        cv2.putText(image, str(i), pt,  cv2.FONT_HERSHEY_PLAIN, 2, (255,0,0), 2)


model_path = 'models/object_detection_3d_chair_1stage.tflite'
# img_path = 'imgs/chairs.jpg'
# img_path = '/sdcard/yoline/objectron-3d-object-detection-openvino/resources/output.jpg'
inShape =[1*640*480*3*4,]
outShape= [1*40*30*1*4, 1*40*30*16*4, 1*160*120*4*4]
    
print('gpu:',aidlite.ANNModel(model_path,inShape,outShape,4,0))

cap=cvs.VideoCapture(0)
while True:
    # img_org = cv2.imread(img_path)
    img_ori=cvs.read()
    if img_ori is None:
        continue
    # img_ori=cv2.flip(img_ori,1)
    
    img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (480, 640)).astype(np.float32)
    img = img / 128.0 - 1.0
    img = img[None]
    # print(img.shape, img.dtype)
    aidlite.setTensor_Fp32(img, 480, 640)
    
    start_time = time.time()
    aidlite.invoke()
    
    gpuelapsed_ms = (time.time() - start_time) * 1000
    print('elapsed_ms invoke:',gpuelapsed_ms)
    cvs.setLbs('elapsed_ms invoke:'+str(gpuelapsed_ms))
    
    
    hm = aidlite.getTensor_Fp32(0)
    displacements = aidlite.getTensor_Fp32(1)
    print(hm.shape, displacements.shape)
    objs = decode(hm, displacements, threshold=0.7)
    for obj in objs:
        draw_box(img_ori, obj)
    cvs.imshow(img_ori)
    
