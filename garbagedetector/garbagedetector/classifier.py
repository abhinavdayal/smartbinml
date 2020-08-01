import caffe
import os
from PIL import Image
import numpy as np
import cv2
import time
from . import S3utils
from django.conf import settings
import datetime

def prepareNet():
	proto_data = open(settings.MEAN_FILE, "rb").read()
	a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
	mean  = caffe.io.blobproto_to_array(a)[0]
	net = caffe.Net(settings.DEPLOY_FILE,settings.CAFFE_MODEL,caffe.TEST)
	return (mean, net)

mean, net = prepareNet()

def resizeForFCN(image,size):
    w,h = image.size
    if w<h:
        return image.resize((int(227*size),int((227*h*size)/w)))
    else:
        return image.resize((int((227*w*size)/h),int(227*size)))
    
def getSegmentedImage(test_image, probMap,thresh):
    kernel = np.ones((6,6),dtype=np.uint8)
    wt,ht = test_image.size
    out_bn = np.zeros((ht,wt),dtype=np.uint8)
    
    for h in range(probMap.shape[0]):
                for k in range(probMap.shape[1]):
                    if probMap[h,k] > thresh:
                        x1 = h*62 #stride 2 at fc6_gb_conv equivalent to 62 pixels stride in input
                        y1 = k*62
                        for hoff in range(x1,227+x1):
                            if hoff < out_bn.shape[0]:
                                for koff in range(y1,227+y1):
                                    if koff < out_bn.shape[1]:
                                        out_bn[hoff,koff] = 255
    edge = cv2.Canny(out_bn,200,250)
    box = cv2.dilate(edge,kernel,iterations = 3)
    
    or_im_ar = np.array(test_image)
    or_im_ar[:,:,1] = (or_im_ar[:,:,1] | box)
    or_im_ar[:,:,2] = or_im_ar[:,:,2] * box + or_im_ar[:,:,2]
    or_im_ar[:,:,0] = or_im_ar[:,:,0] * box + or_im_ar[:,:,0]
    
    return Image.fromarray(or_im_ar)
    
    
def getPredictionsFor(image_files):
    size = 4
    thresh = 0.999

    classifications = []
    #print(len(image_files))
    for i in range(len(image_files)):
            test_image = resizeForFCN(image_files[i],size)
            #print(test_image)
            in_ = np.array(test_image,dtype = np.float32)
            #print(in_)
            in_ = in_[:,:,::-1]
            in_ -= np.array(mean.mean(1).mean(1))
            in_ = in_.transpose((2,0,1))

            net.blobs['data'].reshape(1,*in_.shape)
            net.blobs['data'].data[...] = in_
            net.forward()
            
            probMap =net.blobs['prob'].data[0,1]
            #print(names[i]+'...',)
            if len(np.where(probMap>thresh)[0]) > 0:
                classifications.append("Garbage!")
                #print('Garbage!')
            else:
                classifications.append("Not Garbage!")
                #print('Not Garbage!')
            
            out_ = getSegmentedImage(test_image, probMap,thresh)
            filepath = os.path.join(settings.OUTPUT_FOLDER,
                                    datetime.datetime.utcnow().strftime('%Y%m%d%H%M%S%f')) + '.jpg'
            out_.save(filepath)
            
    return {'classification': classifications, 'image': S3utils.S3Connection.upload(filepath)}

