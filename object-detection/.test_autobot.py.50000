import android, time, sys

droid = android.Android()



title = "AidBot Laucher..." 
sstr = "initing and Loading models..."
# droid.dialogCreateHorizontalProgress(title,sstr,100) 
# droid.dialogShow() 
	
# for x in range(0,50):
#     time.sleep(0.001) 
#     droid.dialogSetCurrentProgress(x) 


from cvs import *

import os
import time
# import cv2
# droid.dialogSetCurrentProgress(60) 
import numpy as np
# import tensorflow as tf
# from tflite_runtime.interpreter import Interpreter
import aidlite_gpu
aidlite=aidlite_gpu.aidlite(3)

# droid.dialogSetCurrentProgress(70) 
# droid.dialogDismiss()

from utils.ssd_mobilenet_utils import *

# from track import KalmanCentroidTrack, KalmanBBoxTrack, ParticleTrack
# from tracker import Tracker
from utils.SimpleTracker import SimpleTracker

# from cvp import *



def run_detection(input_shape,image):
    # Run model: start to detect
    # Sets the value of the input tensor.
    aidlite.setTensor_Fp32(image,input_shape[1],input_shape[1])
    # tflite.setTensor_Int8(image,input_shape[1],input_shape[1])
    # Invoke the interpreter.
    aidlite.invoke()
    
    # print(input_details.shape,output_details.shape)

    # get results
    boxes = aidlite.getTensor_Fp32(0)
    classes = aidlite.getTensor_Fp32(1)
    scores = aidlite.getTensor_Fp32(2)
    num = aidlite.getTensor_Fp32(3)

    box=boxes.reshape((10,4))
    # print("boxes",box)
    # print("scores",scores)
    # print("classes",classes)
    # print("num",num)
    
    box, scores, classes = np.squeeze(box), np.squeeze(scores), np.squeeze(classes + 1).astype(np.int32)
    
    # print("boxes1",boxes)
    # print("scores1",scores)
    # print("classes1",classes)
    # print("num1",num)
    
    out_scores, out_boxes, out_classes = non_max_suppression(scores, box, classes)

    # Print predictions info
    #print('Found {} boxes for {}'.format(len(out_boxes), 'images/dog.jpg'))
            
    return out_scores, out_boxes, out_classes

def image_object_detection(interpreter, colors):
    image = cv2.imread('images/dog.jpg')
    image_data = preprocess_image_for_tflite(image, model_image_size=300)
    out_scores, out_boxes, out_classes = run_detection(image_data, interpreter)

    # Draw bounding boxes on the image file
    result = draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    # Save the predicted bounding box on the image
    cv2.imwrite(os.path.join("out", "ssdlite_mobilenet_v2_dog.jpg"), result, [cv2.IMWRITE_JPEG_QUALITY, 90])

def real_time_object_detection(droid,speed,input_shape, colors,class_names):
    cap = cvs.VideoCapture(1)
    # camera=cvp.VideoCapture(4,0)
    leftControl=0.0
    rightControl=0.0
    state=True
    # ret = "前"
    tracker = SimpleTracker(max_lost=3)  # Initialize tracker
    threshold=0.6
    while True:
        start = time.time()
        frame = cap.read() 
        ret=droid.getSpeechRecognizerResult().result
        print("===================================")
        print(ret)
        if frame is not None:
            frame=cv2.resize(frame,(640,480))
            # image_data = preprocess_image_for_tflite(frame, model_image_size=300)
            
            image_data = preprocess_image_for_tflite_uint8(frame, model_image_size=300)

            # # print(image_data)
            out_scores, out_boxes, out_classes = run_detection(input_shape,image_data)
            # # print("out_boxes",out_boxes)
            # # print("out_scores",out_scores)
            # # print("out_classes",out_classes)
            
            end = time.time()
            
            detections_bbox = np.array([[out_boxes[i][0], out_boxes[i][1], out_boxes[i][2], out_boxes[i][3]] for i in range(len(out_boxes)) if
                out_classes[i] == 1 and out_scores[i] > threshold and out_boxes[i][3]-out_boxes[i][1]>0.1]).reshape(-1, 4)
            
            objects = tracker.update(detections_bbox)
            
            leftControl,rightControl=(0.0,0.0)
            
            buf = droid.UsbRead()
            # print('buf:',buf,frame.shape)
            
            # print("objects:",objects)
            h,w,_ = frame.shape
            
            for (objectID, centroid) in objects.items():
                text = "ID {}".format(objectID)
                print(text,centroid)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            1,  (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, (centroid[0], centroid[1]), 4,  (0, 255, 0), -1)            
                
                # centerX=(left+right)/2
                centerX=centroid[0]
                
                x_pos_norm = 1.0 - 2.0 * centerX / w;
                
                if x_pos_norm >0:
                    leftControl = 1.0
                    rightControl = 1.0 + x_pos_norm
                else :
                    leftControl = 1.0 - x_pos_norm
                    rightControl = 1.0
                break;
            
            #Draw bounding boxes on the image file
            result = draw_boxes_and_turn(frame, out_scores, out_boxes, out_classes, class_names, colors)
            if ret is None or "前" in ret:
                state=True
            elif "停" in ret or "stop" in ret:
                state=False
            if state:
                droid.sendControlToVehicle(int(leftControl*speed),int(rightControl*speed))
            else:
                droid.sendControlToVehicle(0,0)
            

            # fps
            t = end - start
            fps  = "time: {:.2f}".format(t*1000)
            cv2.putText(frame, fps, (10, 30),
		                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
            lbs = 'Average FPS: '+ str(1 /t)
            cvs.setLbs(lbs)
            
            cvs.imshow(frame)
            if cv2.waitKey(5) & 0xFF == ord('q'):
                # cvp.close()
                break
            
    # camera.release()
    # cv2.destroyAllWindows()

def main():
    # Load TFLite model and allocate tensors.
    # import tflite_runtime.interpreter as tflite
    
    inShape =[1 * 300 * 300 *3,]
    outShape= [1 * 10*4*4,1*10*4,1*10*4,1*4]
    model_path="model_data/ssdlite_mobilenet_v3.tflite"
    # model_path="model_data/model_integer_quant.tflite"
    
    print('gpu:',aidlite.FAST_ANNModel(model_path,inShape,outShape,4,0))#4表示4个线程，0表示gpu，-1表示cpu，1表示NNAPI
    
    # interpreter = Interpreter(model_path="model_data/ssdlite_mobilenet_v3.tflite",num_threads=4)
    # interpreter.allocate_tensors()

    # Get input and output tensors.
    # input_details = interpreter.get_input_details()
    # output_details = interpreter.get_output_details()
    input_shape=[300,300]


    # print("=======================================")
    # print("input :", str(input_details))

    # print("ouput :", str(output_details))
    # print("=======================================")

    # label
    class_names = read_classes('model_data/coco_classes.txt')
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
            
    #image_object_detection(interpreter, colors)
    
    print("connecting to usb bot")
    r=droid.connectUsb().result
    if r:
        droid.sendIndicatorToVehicle(0)
        speed=92;#range(1,250)
        p=droid.requestRecordPermission().result
        if p:
            flag=droid.registerSpeechRecognizer().result
            # print('语音识别注册是否成功'+flag)
            if flag:
                   droid.startSpeechRecognize()
        real_time_object_detection(droid,speed,input_shape, colors,class_names)
        
        
    else:
        ret=droid.getAlertDialog().result
        if ret:
              droid.sendIndicatorToVehicle(0)
              speed=92;#range(1,250)
              real_time_object_detection(droid,speed,input_shape, colors,class_names)

main()






