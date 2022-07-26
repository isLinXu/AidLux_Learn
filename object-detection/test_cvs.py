from cvs import *
# import facerecognition
import os
import time
import cv2
import numpy as np
import tensorflow as tf
from utils.ssd_mobilenet_utils import *
from cvp import *


def run_detection(input_details,image, interpreter,output_details):
    # Run model: start to detect
    # Sets the value of the input tensor.
    interpreter.set_tensor(input_details[0]['index'], image)
    # Invoke the interpreter.
    interpreter.invoke()

    # get results
    boxes = interpreter.get_tensor(output_details[0]['index'])
    classes = interpreter.get_tensor(output_details[1]['index'])
    scores = interpreter.get_tensor(output_details[2]['index'])
    num = interpreter.get_tensor(output_details[3]['index'])

    boxes, scores, classes = np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes + 1).astype(np.int32)
    out_scores, out_boxes, out_classes = non_max_suppression(scores, boxes, classes)

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

def real_time_object_detection(input_details,interpreter, colors,output_details,class_names):
    camera = cvs.VideoCapture(0)
    # camera=cvp.VideoCapture(4,0)

    while True:
        start = time.time()
        frame = camera.read() 

        if frame is not None:
            # image_data = preprocess_image_for_tflite(frame, model_image_size=300)
            image_data = preprocess_image_for_tflite_uint8(frame, model_image_size=300)

            # print(image_data)
            out_scores, out_boxes, out_classes = run_detection(input_details,image_data, interpreter,output_details)
            # Draw bounding boxes on the image file
            result = draw_boxes(frame, out_scores, out_boxes, out_classes, class_names, colors)
            end = time.time()

            # fps
            t = end - start
            fps  = "Fps: {:.2f}".format(1 / t)
            cv2.putText(result, fps, (10, 30),
		                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
            cvs.imshow(frame)
            if cv2.waitKey(30) & 0xFF == ord('q'):
                # cvp.close()
                break
            
    # camera.release()
    # cv2.destroyAllWindows()

def main():
    # Load TFLite model and allocate tensors.
    # import tflite_runtime.interpreter as tflite
    interpreter = tf.lite.Interpreter(model_path="model_data/ssdlite_mobilenet_v3.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()


    print("=======================================")
    print("input :", str(input_details))

    print("ouput :", str(output_details))
    print("=======================================")

    # label
    class_names = read_classes('model_data/coco_classes.txt')
    # Generate colors for drawing bounding boxes.
    colors = generate_colors(class_names)
            
    #image_object_detection(interpreter, colors)
    real_time_object_detection(input_details,interpreter, colors,output_details,class_names)



class MyApp(App):
    def __init__(self, *args):
        super(MyApp, self).__init__(*args)
    
    def idle(self):
        #idle function called every update cycle
        self.lbl.set_text(cvs.getLbs())
        pass
        
    def main(self):
        #creating a container VBox type, vertical (you can use also HBox or Widget)
        main_container = VBox(width=360, height=680, style={'margin':'0px auto'})
        
        self.aidcam = OpencvVideoWidget(self, width=340, height=480)
        self.aidcam.style['margin'] = '10px'
        
        self.aidcam.identifier="myimage_receiver"
        main_container.append(self.aidcam)
        
        self.lbl = Label('This show fps!', width=360, height=30,  margin='50px',)
        main_container.append(self.lbl)

        return main_container
    
    

if __name__ == '__main__':

    initcv(main)
    startcv(MyApp)


