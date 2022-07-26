from cvs import *
import aidlite_gpu
aidlite=aidlite_gpu.aidlite(1)


labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

w=224
h=224
input_shape=[w,h]
inShape =[w*h*3*4,]
outShape= [1 * w*h,]
model_path="models/mobilenet_ssd_voc_ncnn"
print('cpu:',aidlite.ANNModel(model_path,inShape,outShape,4,0))




while True:
    
    print('tnn: start set')
    img=cvs.imread('tiger_cat_224.jpg')
    # print('=====',img.dtype, img.shape)
    input =img.astype(np.float32)
    input = (input-127.5) / 127.5
    aidlite.setInput_Float32(input,w,w)
    # print('===================')
    start_time = time.time()
    print('tnn: start invoke')
    aidlite.invoke()
    
    t = (time.time() - start_time)
    # print('elapsed_ms invoke:',t*1000)
    lbs = 'Fps: '+ str(int(1/t))+" ~~ Time:"+str(t*1000) +"ms"
    
    print('tnn: start get')
    pred = aidlite.getOutput_Float32(0)
    print('======', pred)

    label_id, prob = int(pred[0]),pred[1]
    
    p1 = (int(pred[2]*w+0.5), int(pred[3]*w+0.5))
    p2 = (int(pred[4]*w+0.5), int(pred[5]*w+0.5))
            
    
    cvs.rectangle(img, p1,p2, (0, 0, 255) , 3, 1)
    
    print ('result:',label_id,prob,labels[label_id])
    lbs+=" result:"+labels[label_id]+":%f "%prob
    cvs.setLbs(lbs)
    
    cvs.imshow(img)
    # sleep(500)
    
    


