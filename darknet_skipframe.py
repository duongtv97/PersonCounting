from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import sys

import warnings

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet     #IMPORTANT , import Deep Cosine Extractor
from deep_sort.detection import Detection as ddet


from deep_sort.feature_extractor_MGN import Extractor   #IMPORTANT, import MGN Extractor 

warnings.filterwarnings('ignore')

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:
        if detection[0].decode()!="person": continue
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() ,
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None

def get_features(boxes, ori_img, extractor,w , h):
        features = []
        for box in boxes:
            x1,y1,x2,y2 = xywh_to_xyxy(box, w, h)
            im = ori_img[y1:y2,x1:x2]
            feature = extractor(im)[0]
            #print ("shape of feature: ", feature.shape)
            features.append(feature)
        if len(features):
            features = np.stack(features, axis=0)
        else:
            features = np.array([])
        return features

def xywh_to_xyxy(bbox_xywh, width, height):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2),0)
        x2 = min(int(x+w/2),width-1)
        y1 = max(int(y-h/2),0)
        y2 = min(int(y+h/2),height-1)
        return x1,y1,x2,y2
                
def YOLO():

    global metaMain, netMain, altNames
    configPath = "./cfg/yolov3-spp.cfg"
    weightPath = "./yolov3-spp.weights"
    metaPath = "./cfg/coco.data"
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    
    #create overlay for computing heatmap 
    overlay_empty = np.zeros((608,608,1), np.uint8)
    
    
    cap = cv2.VideoCapture("video_test/01.avi")         #IMPORTANT 
    cap.set(3, 1280)
    cap.set(4, 720)
    #print ("width", cap.get(3))
    #print ("height", cap.get(4))
    """
    out = cv2.VideoWriter(
        "video_output/modded_2-5f_124_pytorch_MGN_thres00965_dthres0375_x30y160_age7000_cbox20-10w-5h-160.mp4", cv2.VideoWriter_fourcc(*"MP4V"), 20,
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")
    """
    out = cv2.VideoWriter(
        "txt_output_n50_0960/01.avi", cv2.VideoWriter_fourcc(*"MP4V"), 20,   #IMPORTANT
        (darknet.network_width(netMain), darknet.network_height(netMain)))
    print("Starting the YOLO loop...")
    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
                                    
    # ================= Definition of the parameters
    max_cosine_distance = 0.096   #IMPORTANT PARAMETER
    maxAge = 100                  #IMPORTANT PARAMETER
    nn_budget = None
    nms_max_overlap = 1.0
    
    # ================= Models location  #IMPORTANT 
    #model_filename = 'model_data/mars-small128.pb'  #Deep Cosine Metric model
    model_filename = 'model_data/model_MGN.pt'      #MGN model                 
    
    
    # ========================= Init MGN feature extractor
    extractor = Extractor(model_filename, use_cuda=True)     #IMPORTANT 
    
    # ========================= Init Deep Cosine feature extractor
    #encoder = gdet.create_box_encoder(model_filename,batch_size=1)    #IMPORTANT 
    
    
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric, max_age = maxAge)
    fps = 0.0
    counts = 1
    ids = dict()
    b = set()
    
    #fi = open("txt_output_n50_0960/01.txt","w")
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        if ret != True : break
        if counts % 1 ==0: 
            print("========FRAME========: " , counts)
            
            frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb,
                                       (darknet.network_width(netMain),
                                        darknet.network_height(netMain)),
                                       interpolation=cv2.INTER_LINEAR)
            height, width, channels = frame_resized.shape
            #print ("Shape of frame: ",frame_resized.shape)
            darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())
    
            boxs = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.375)
            boxes= []
            
            for detection in boxs:
                if detection[0].decode()!="person": continue
                x, y, w, h = detection[2][0],\
                detection[2][1],\
                detection[2][2],\
                detection[2][3]
                #xmin = int(round(float(x) - (float(w) / 2)))
                #ymin = int(round(float(y) - (float(h) / 2)))
                xmin, ymin, xmax, ymax = convertBack(int(x), int(y), int(w), int(h))
                
                #========================= generate HEATMAP values ====================
                x_heat = int(round((xmin+xmax)/2))
                overlay_empty[int(ymax)-10:int(ymax)+10,x_heat-10:x_heat+10]+=5
                #======================================================================
                
                
                cv2.rectangle(frame_resized, (xmin,ymin), (xmax,ymax), (255,0, 255), 1)
                
                if (xmin > 100 and ymin > 10) and ( xmax < width - 5 and ymax < height - 120):
                    if (xmax -xmin ) > 30 and (ymax - ymin) > 160:
                        boxes.append([xmin,ymin,w,h])
                        
            pt1=(100,10)
            pt2=(width-5,height-120)
            cv2.rectangle(frame_resized, pt1, pt2, (0, 255, 0), 1)
            
            # ==================== EXTRACT FEATURES ==================== #IMPORTANT
            #features = encoder(frame_resized,boxes)      #Deep Cosine Metric extractor
            features = get_features(boxes, frame_resized, extractor, width, height)     #MGN extractor
            # ============================================================
            
            
            detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxes, features)]
            
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            
            #update tracker
            tracker.predict()
            tracker.update(detections)
            
            # ====================== Fix the ID jumping problem ======================
            if counts == 1:
                count = 1
                for track in tracker.tracks:    
                    ids[track.track_id] = count
                    count += 1
            else: 
                count = 0
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    if track.track_id in ids: continue 
                    if len(list(ids.values()))!=0:
                        count = max(list(ids.values()))
                    ids[track.track_id] = count + 1
            #print (ids)
            # =========================================================================    
            
            cv2.putText(frame_resized, "FrameID: " + str(counts),(350,30),0, 5e-3 * 150, (255,255,0),2)
            
            for track in tracker.tracks:            
                if not track.is_confirmed() or track.time_since_update > 1:
                    continue 
                #print ("=======ID at the frame=======: ", track.track_id)
                bbox = track.to_tlbr()
                #heat map 
                
                #print("x heat coordinate: ", x_heat)
                
                cv2.rectangle(frame_resized, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,0), 2)
                if track.track_id in ids : 
                    cv2.putText(frame_resized, str(ids[track.track_id]),(int(bbox[2]), int(bbox[3])),0, 5e-3 * 250, (255,255,0),2)
                    if counts%2==0:
                        fi.write(str(counts)+" "+str(int(bbox[2]))+" "+str(int(bbox[3]))+" "+str(ids[track.track_id])+" "+"\n")
                b.add(track.track_id)
                #if (np.allclose(overlay_empty[int(bbox[3])-5:int(bbox[3])+5,x_heat-5:x_heat+5,0],255)): continue
                
            #print (b)
            if len(list(ids.values()))!=0:
                text = "Num people: {}".format(max(list(ids.values())))
                cv2.putText(frame_resized, text,(1, 30),0, 5e-3 * 200, (255,0,0),2)
            overlay_empty[np.where((overlay_empty[:,:,0] > 200))] = 230    
            """
            for det in detections:
                bbox = det.to_tlbr()
                
                
                cv2.rectangle(frame_resized,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)
            """
            
            # ============== BLENDING HEATMAP VALUES TO FRAME ==============
            im_color = cv2.applyColorMap(overlay_empty, cv2.COLORMAP_JET)
            im_color = cv2.blur(im_color,(10,10))
            #im_color[:,:,0]=0
                    
            image = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)   
            #image = cvDrawBoxes(detections, frame_resized)
            heat = cv2.addWeighted(image, 0.7, im_color , 0.3, 0)
            # ===============================================================
            
            
            #counts = counts + 1
            out.write(heat)
            print("fps: ",1/(time.time()-prev_time))
            print ("\n")
            #fps  = ( fps + (1./(time.time()-prev_time)) ) / 2
            #print("fps= %f"%(fps))
            #cv2.imshow('Demo', image)
            #cv2.imshow('', heat)
            #cv2.imshow('', cv2.resize(image,(1024,600)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        counts = counts + 1
        fps = fps + (1/(time.time()-prev_time))
    print ("average fps: ", fps/counts)    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
def parse_args():
    parser = argparse.ArgumentParser(description="Input Parameter") 
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--max_cosine_distance",
        default = 0.3)
    parser.add_argument(
        "--max_age",
        default = 50)    
if __name__ == "__main__":
    YOLO()
