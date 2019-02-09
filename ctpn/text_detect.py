import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from lib.utils.timer import Timer
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import  test_ctpn
from lib.networks.factory import get_network
from lib.text_connector.detectors import TextDetector
from lib.text_connector.text_connect_cfg import Config as TextLineCfg

import re
from math import fabs,cos,sin,radians,pi,atan
import argparse
parser = argparse.ArgumentParser(description='根据文件名，旋转并保存目录下的图片')
parser.add_argument('--imgs2rtt', type=str, default = "./imgs2rtt")
parser.add_argument('--imgs2rtted', type=str, default = "./imgs2rtted")

args = parser.parse_args()
imgs2rtt = args.imgs2rtt
imgs2rtted = args.imgs2rtted

def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

def load_tf_model():
    # load config file
    cfg.TEST.checkpoints_path = './ctpn/checkpoints'

    # init session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    sess = tf.Session(config=config)

    # load network
    net = get_network("VGGnet_test")

    # load model
    print('Loading network {:s}... '.format("VGGnet_test"))
    saver = tf.train.Saver()
    try:
        ckpt = tf.train.get_checkpoint_state(cfg.TEST.checkpoints_path)
        print('Restoring from {}...'.format(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('done')
    except:
        raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)

    return sess, net

sess, net = load_tf_model()

def ctpn(img):
    timer = Timer()
    timer.tic()

    img, scale = resize_im(img, scale=TextLineCfg.SCALE, max_scale=TextLineCfg.MAX_SCALE)
    scores, boxes = test_ctpn(sess, net, img)

    textdetector = TextDetector()
    boxes = textdetector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    timer.toc()
    print("\n----------------------------------------------")
    print(('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0]))

    return scores, boxes, img, scale

def draw_boxes(img, boxes, scale):
    box_id = 0
    img = img.copy()
    text_recs = np.zeros((len(boxes), 8), np.int)
    for box in boxes:
        if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
            continue

        if box[8] >= 0.8:
            color = (255, 0, 0)  # red
        else:
            color = (0, 255, 0)  # green

        cv2.line(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[0]), int(box[1])), (int(box[4]), int(box[5])), color, 2)
        cv2.line(img, (int(box[6]), int(box[7])), (int(box[2]), int(box[3])), color, 2)
        cv2.line(img, (int(box[4]), int(box[5])), (int(box[6]), int(box[7])), color, 2)

        for i in range(8):
            text_recs[box_id, i] = box[i]

        box_id += 1

    img = cv2.resize(img, None, None, fx=1.0/scale, fy=1.0/scale) #,  interpolation=cv2.INTER_LINEdstAR
    return text_recs, img

def text_detect(img):
    scores, boxes, img, scale = ctpn(img)
    print(boxes)
    text_recs, img_drawed = draw_boxes(img, boxes, scale)
    return text_recs, img_drawed, img

#1 计算所有找到的box的倾斜度数
#2 对于大于3个box的情况，去掉最小和最大的度数（opencv的旋转是逆时针度数0-360）
#3 取剩下的box的平均度数beta
#4 按beta度数旋转
#5 继续回到步骤1，按序处理N遍，默认N==2

#[5.10013031e+02 5.05802063e+02 6.72000000e+02 5.00832306e+02
#  5.12000000e+02 5.70566895e+02 6.73987000e+02 5.65597168e+02
#  6.92928195e-01]math.asin(x)
def rotateImg4TextDetection(img,N=5):
    # print(img.shape)
    # 序处理N遍，默认N==5
    degreeTotal = 0
    for i in range(N):
        # print(i)
        scores, boxes, img, scale = ctpn(img)
        # print(boxes)
        boxCount = len(boxes)
       # 计算所有找到的box的倾斜度数
        betaTotal = 0
        for box in boxes:
            xlefttop = box[0]
            ylefttop = box[1]

            xrighttop = box[2]
            yrighttop = box[3]

            beta = calculateAngle(xlefttop,ylefttop,xrighttop,yrighttop)
            beta = 90-beta # 90度数
            betaTotal = betaTotal + beta
            #print("beta=",beta)
        # print(betaTotal)
        # 大于90度，表示左低右高；旋转 = 360 +（90 - beta）
        # 小于90度，表示左高右低；旋转 = 90 - beta
        betaAvg = betaTotal/len(boxes)
        degree = 0
        if betaAvg>0: # 小于90度
            degree = betaAvg
        elif betaAvg<0: #大于90度
            degree = 360+betaAvg
        # accumulate degree to degreeTotal
        degreeTotal = degreeTotal+degree

        # rows, cols,_ = img.shape
        # M = cv2.getRotationMatrix2D(((cols - 1) / 2.0, (rows - 1) / 2.0), angle, 1)
        # imgRotation = cv2.warpAffine(img, M, (cols, rows) ,borderValue=(255,255,255))
        height_ori, width_ori = img.shape[:2]
        height_1 = int(width_ori * fabs(sin(radians(degree))) + height_ori * fabs(cos(radians(degree))))
        width_1 = int(height_ori * fabs(sin(radians(degree))) + width_ori * fabs(cos(radians(degree))))
        matRotation = cv2.getRotationMatrix2D((width_ori / 2, height_ori / 2), degree, 1)
        matRotation[0, 2] += (width_1 - width_ori) / 2
        matRotation[1, 2] += (height_1 - height_ori) / 2
        imgRotation = cv2.warpAffine(img, matRotation, (width_1, height_1), borderValue=(255, 255, 255)) #
        img = imgRotation
    return img, degreeTotal

# 计算方位角函数
def calculateAngle( x1,  y1,  x2,  y2):
    angle = 0.0;
    dx = x2 - x1
    dy = y2 - y1
    if  x2 == x1:
        angle = pi / 2.0
        if  y2 == y1 :
            angle = 0.0
        elif y2 < y1 :
            angle = 3.0 * pi / 2.0
    elif x2 > x1 and y2 > y1:
        angle = atan(dx / dy)
    elif  x2 > x1 and  y2 < y1 :
        angle = pi / 2 + atan(-dy / dx)
    elif  x2 < x1 and y2 < y1 :
        angle = pi + atan(dx / dy)
    elif  x2 < x1 and y2 > y1 :
        angle = 3.0 * pi / 2.0 + atan(dy / -dx)
    return (angle * 180 / pi)

def rotateWriteImgByCv2(fromImg,toImg,rotateDegree):
    img = cv2.imread(fromImg, 1)
    # rows, cols, channel = img.shape
    # M = cv2.getRotationMatrix2D((cols / 2, rows / 3), rotateDegree, 1)
    # dst = cv2.warpAffine(img, M, (cols, rows))

    height_ori, width_ori, channel = img.shape
    height_1 = int(width_ori * fabs(sin(radians(rotateDegree))) + height_ori * fabs(cos(radians(rotateDegree))))
    width_1 = int(height_ori * fabs(sin(radians(rotateDegree))) + width_ori * fabs(cos(radians(rotateDegree))))
    matRotation = cv2.getRotationMatrix2D((width_ori / 2, height_ori / 2), rotateDegree, 1)
    matRotation[0, 2] += (width_1 - width_ori) / 2
    matRotation[1, 2] += (height_1 - height_ori) / 2
    dst = cv2.warpAffine(img, matRotation, (width_1, height_1), borderValue=(255, 255, 255))  #

    cv2.imwrite(toImg, dst)

def rotateAllImgInFolder():

    list = os.listdir(imgs2rtt)
    # print(list)
    i=0
    for f in list:
        i = i + 1
        print(i)
        if f.endswith(".jpg"): #
            imgPath = os.path.join(imgs2rtt, f)
            print(imgPath)
            im = Image.open(imgPath)
            img = np.array(im.convert('RGB'))
            _, degree2Rotate = rotateImg4TextDetection(img,7)
            img2RotatePath = os.path.join(imgs2rtted, f)
            rotateWriteImgByCv2(imgPath, img2RotatePath, degree2Rotate)


from PIL import Image
from lib.fast_rcnn.config import cfg_from_file

if __name__ == '__main__':
    cfg_from_file('./ctpn/ctpn/text.yml')
    rotateAllImgInFolder()
# if __name__ == '__main__':
#     cfg_from_file('./ctpn/ctpn/text.yml')
#     filePrefix = './test_images/36972690-5c30645aca86ca1093902db4'
#     fileName = filePrefix+'.jpg'
#     im = Image.open(fileName)
#     img = np.array(im.convert('RGB'))
#
#     _, degree2Rotate = rotateImg4TextDetection(img)
#     toImg = filePrefix+'_rotated4.jpg'
#     rotateWriteImgByCv2(fileName,toImg,degree2Rotate)
#
#     rotatedIm = Image.open(toImg)
#     img_rotated = np.array(rotatedIm.convert('RGB'))
#
#     text_recs, img_drawed, img = text_detect(img_rotated)
#     Image.fromarray(img_drawed).save(filePrefix+'_result4.jpg')


