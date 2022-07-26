from cvs import *
import aidlite_gpu
aidlite=aidlite_gpu.aidlite()


def classify_image(output, top_k=1):
  ordered = np.argpartition(-output, top_k)
  return [(i, output[i]) for i in ordered[:top_k]]

def load_labels(path):
  with open(path, 'r') as f:
    return {i: line.strip() for i, line in enumerate(f.readlines())}

w=224
h=224
input_shape=[w,h]
inShape =[w*h*3*4,]
outShape= [1 * w*h,]
model_path="models/mobilenet_v2-vulkan.pt"
print('gpu:',aidlite.ANNModel(model_path,inShape,outShape,4,1))

labels = load_labels("models/synset_words.txt")
img=cvs.imread('tiger_cat_224.jpg')

while True:
    cvs.imshow(img)
    print('tnn: start set')
    input =img.astype(np.float32)
    input = (input-127.5) / 127.5
    aidlite.setInput_Float32(input,w,w)
    start_time = time.time()
    print('tnn: start invoke')
    aidlite.invoke()
    
    t = (time.time() - start_time)
    # print('elapsed_ms invoke:',t*1000)
    lbs = 'Fps: '+ str(int(1/t))+" ~~ Time:"+str(t*1000) +"ms"
    
    print('tnn: start get')
    pred = aidlite.getOutput_Float32(0)
    result= classify_image(pred)
    label_id, prob = result[0]
    
    print ('result:',label_id,prob,labels[label_id])
    lbs+=" result:"+labels[label_id]+":%f "%prob
    cvs.setLbs(lbs)
    
    


