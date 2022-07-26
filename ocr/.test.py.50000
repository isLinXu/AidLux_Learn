import cv2
import math
import numpy as np
from cvs import *
from Detector import DetectorDecoder
import aidlite_gpu
from PIL import Image, ImageDraw, ImageFont

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=50):
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top - textSize), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return img


def det_process(mat, mean, std):
    image = mat.copy()
    img = cv2.resize(image, (640, 640))
    img = img.astype(np.float32)
    img /= 255.0
    img -= mean
    img /= std
    img = img.transpose(2, 0, 1)[None]
    return img


def rec_process(image):
    img = cv2.resize(image, (100, 32))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img - 127.5
    img /= 127.5
    img = img.transpose(2, 0, 1)[None].astype(np.float32)
    return img


def crop(mat, box):
    img = mat.copy()
    w, h, _ = img.shape

    w_s = w / 640
    h_s = h / 640

    high = box[1][0] - box[0][0]
    wide = box[3][1] - box[0][1]

    o = [box[0][0] * h_s, box[0][1] * w_s]

    image = img[int(o[1]):int(math.ceil(o[1] + wide * w_s)), int(o[0]):int(math.ceil(o[0] + high * h_s))]
    
    return image, o


def get_keyword_str(file):
    key_str = u''
    with open(file, 'r', encoding='utf8') as fin:
        for line in fin:
            key_str += line.replace('\n', '')
        fin.close()
    return key_str


if __name__ == '__main__':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    decode_handel = DetectorDecoder()
    chineseocr_words_file = 'model/ppocr_keys_v1.txt'
    characters = get_keyword_str(chineseocr_words_file)

    # 加载定位模型
    det = aidlite_gpu.aidlite()
    inShape = [1, 3, 640, 640]
    outShape = [1 * 640 * 640 * 4, ]
    model_path = "model/ch_ppocr_mobile_v2.0_det_slim_opt.nb"
    det.ANNModel(model_path, inShape, outShape, 4, 0, 0)

    # 加载检测模型
    rec = aidlite_gpu.aidlite()
    inShape = [1, 3, 32, 100]
    outShape = [1 * 25 * 6625 * 4, ]
    model_path = "model/ch_ppocr_mobile_v2.0_rec_slim_opt.nb"
    rec.ANNModel(model_path, inShape, outShape, 4, 0, 0)
    
    # 加载数据
    cap = cvs.VideoCapture(0)
    while True:
        frame = cap.read()
        if frame is None:
            continue
        # 定位
        draw = Image.fromarray(frame)
        img = det_process(frame, mean, std)
        det.setInput_Float32(img)
        det.invoke()
        pred_0 = det.getOutput_Float32(0)
        out = pred_0.reshape(640, 640)
        box_list = decode_handel(out, 640, 640)
        
        if len(box_list) == 0:
            cvs.imshow(frame)
        else:
            result = []
            for box in box_list:
                img, port = crop(frame, box)
                img = rec_process(img)
                rec.setInput_Float32(img)
                rec.invoke()
                out = rec.getOutput_Float32(0).reshape(-1, 1, 6625)
                char_list = []
                pred_text = out.argmax(axis=2)
                for i in range(len(pred_text)):
                    if pred_text[i] != 0 and (not (i > 0 and pred_text[i - 1] == pred_text[i])) and pred_text[i]<6624:
                        char_list.append(characters[int(pred_text[i]) - 1])
                string = u''.join(char_list)
                result.append(string)
                draw = cv2ImgAddText(draw, string, int(port[0]), int(port[1]), (255, 0, 0), 20)
            frame = np.asarray(draw)
            cvs.imshow(frame)


