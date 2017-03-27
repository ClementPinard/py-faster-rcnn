#!/usr/bin/python
"""
  RCNN with Jumping Sumo

"""

import sys
import os


import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time
from jumpingsumo import JumpingSumo
from commands import moveCmd, jumpCmd, loadCmd, postureCmd, addCapOffsetCmd, setVolumeCmd

# this will be in new separate repository as common library fo robotika Python-powered robots
from apyros.metalog import MetaLog, disableAsserts
from apyros.manual import myKbhit, ManualControlException
from cStringIO import StringIO
from PIL import Image
import _init_paths
import caffe
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
#import matplotlib.pyplot as plt
import scipy.io as sio
import argparse
import datetime 
# Classes that this caffe model can detect
CLASSES = ('__background__',
           'mug','apple','shoe','glasses')
ICONS = []

COLORS = [(-1,-1,-1),(0,0,0),(0,255,0),(255,0,0),(0,0,255)]

g_lastImage = None

lastkey = False
rcnn=True
stats=False
videorecording=False
video=None

# convert raw image into opencv image
def cvImage(image):
    file_jpgdata = StringIO(g_lastImage)

    im = Image.open(file_jpgdata)
    open_cv_image = np.array(im) 
    # Convert RGB to BGR 
    return open_cv_image[:, :, ::-1].copy() 
    
def keepLastImage( frame, debug=False ):
    global lastkey
    #Make sure key press gets some time so leave if the last call was keepLastImage
    if not lastkey:
        return
    global g_lastImage
    g_lastImage = frame[2]
    open_cv_image=cvImage(g_lastImage)

    global rcnn
    if rcnn:
        # Detect objects in image
        detect(net, open_cv_image)
    global videorecording
    if videorecording:
        video.write(open_cv_image)
    cv2.imshow('image',open_cv_image)
    cv2.waitKey(30)
    lastkey=False



def runSumo( robot ):

    DEG_STEP = 30
    robot.setVideoCallback( keepLastImage )
    im = None
    global g_lastImage
    global lastkey
    global rcnn
    global stats
    global video
    global videorecording
    while True:
        lastkey=True
        dobreak = False
        move = False
        k= cv2.waitKey(30)
        if k==27:
            if videorecording:
                video.release()
            dobreak=True
        elif k==ord(' '):
            robot.update( cmd=jumpCmd(1), ackRequest=True )
        elif k==ord('r'):
            rcnn= not rcnn  
            if rcnn:
                print 'rcnn now on'
            else:
                print 'rcnn now off' 
        elif k==ord('s'):
            stats= not stats  
            if stats:
                print 'stats now on'
            else:
                print 'stats now off' 
        elif k==ord('v'):
            videorecording= not videorecording
            if videorecording:
                fps=5
                open_cv_image=cvImage(g_lastImage)
                height , width , layers =  open_cv_image.shape
                today = datetime.datetime.now()
                filename= "img{:%Y-%m-%d-%H%M%S}".format(today)
                video = cv2.VideoWriter(os.path.join('ardrone','video',filename+'.avi'),-1,fps,(width,height))
                video.write(open_cv_image)
                print 'Video recording on'
            else:
                video.release()
                print 'Video recording stopped'
        elif k==ord('w'):
            open_cv_image=cvImage(g_lastImage)
            today = datetime.datetime.now()
            filename= "img{:%Y-%m-%d-%H%M%S}".format(today)
            cv2.imwrite(os.path.join('ardrone','images',filename+'.png'),open_cv_image) 
            print 'Screenshot saved to '+os.path.join('ardrone','images',filename+'.png')
        elif k==ord('j') or k==ord('i') or k==ord('k') or k==ord('l'):
            move= k		
        if dobreak:
            break
        if move:
            MOVE_STEP=0
            if move==ord('j'):
                DEG_STEP = -25
            elif move==ord('l'):
                DEG_STEP = 25
            elif move==ord('i'): 
                DEG_STEP = 0
                MOVE_STEP = 25
            elif move==ord('k'): 
                DEG_STEP = 0
                MOVE_STEP = -25
        
            if MOVE_STEP!=0:
                robot.update( cmd=moveCmd(MOVE_STEP, 0) )
                #robot.update( cmd=moveCmd(0,0) )
            if DEG_STEP!=0:
                robot.update( cmd=addCapOffsetCmd(math.radians(DEG_STEP)) )
            print "Battery:", robot.battery
            #robot.wait(0.1)
        else:
            robot.update( cmd=moveCmd(0,0) )
        #robot.wait(0.1)
 
def vis_detections(im, class_name, cls_ind, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        # Draw box around object
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), COLORS[cls_ind], 2)
        # Write class name
        cv2.putText(im,'{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1]) - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[cls_ind],2)
        # Draw icon top left
        im[bbox[1]:bbox[1]+16,bbox[0]:bbox[0]+16] = ICONS[cls_ind]


def detect(net, im):
    """Detect object classes in an image using pre-computed object proposals."""

    # Detect all object classes and regress object bounds
    global stats
    if stats:
        timer = Timer()
        timer.tic()
    scores, boxes = im_detect(net, im)
    if stats:
        timer.toc()
        print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, cls_ind, dets, thresh=CONF_THRESH)



if __name__ == "__main__":
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    cpu_mode=False
    gpu_id=0
    prototxt = os.path.join(cfg.MODELS_DIR, 
                            'VGG_CNN_M_1024','faster_rcnn_end2endmasg', 'my.pt')

    caffemodel = os.path.join('data',
                              'forthtemple_faster_rcnnmasgv_iter_20000.caffemodel')
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_forthtemple_models.sh from root of your project?').format(caffemodel))

    if cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(gpu_id)
        cfg.GPU_ID = gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    ICONS.append(None)
    #COLORS.append(None)  
    for cls in CLASSES:
        file=os.path.join('ardrone','icons','small',cls+'.png')
        
        if os.path.isfile(file):
            img=cv2.imread(file) #, cv2.IMREAD_UNCHANGED)
            #print cls
            '''(B, G, R, A) = cv2.split(img)
            B = cv2.bitwise_and(B, B, mask=A)
            G = cv2.bitwise_and(G, G, mask=A)
            R = cv2.bitwise_and(R, R, mask=A)
            img = cv2.merge([B, G, R, A])
            '''
            img=cv2.resize(img, (16,16)) 
            ICONS.append(img)
            #COLOR.append()
    metalog=None
    print('j=turn left, i=forward, l=turn right, k=back, spc jump')
    print('r=toggle rcnn on and off, s=stats on and off, w=save screen, v=toggle video record')
    robot = JumpingSumo( metalog=metalog )   
    robot.update( cmd=setVolumeCmd(0) )
    #setVolumeCmd( volume )	
    runSumo( robot )
    print "Battery:", robot.battery

# vim: expandtab sw=4 ts=4 

