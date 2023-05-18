import cv2 as cv
from ultralytics import YOLO
import numpy as np
import torch
import glob
import os




def cut_area(cam_x1, cam_y1, cam_x2, cam_y2, scr_x1, scr_y1, scr_x2, scr_y2):
    cut_x1 = max(cam_x1, scr_x1)
    cut_x2 = min(cam_x2, scr_x2)
    if cut_x1 > cut_x2:
        return False, []
    cut_y1 = max(cam_y1, scr_y1)
    cut_y2 = min(cam_y2, scr_y2)
    if cut_y1 > cut_y2:
        return False, []
    return True, [cut_x1, cut_x2, cut_y1, cut_y2]

def f_S(x1, y1, x2, y2):
    return (x2 - x1) * (y2 - y1)

def scaling_f(y):
    if y == 0:
        return 2
    return y + 1.5

def S_inbound(x1, y1, x2, y2, bound1, bound2, coord):
    if coord:
        return S(x1, max(y1, bound1), x2, min(y2, bound2))
    return S(max(x1, bound1), y1, min(x2, bound2), y2)

def union(box1, box2):
    ret, area = cut_area(box1[0], box1[1], box1[2], box1[3], box2[0], box2[1], box2[2], box2[3])
    if ret:
        return ret, f_S(area[0], area[2], area[1], area[3])
    else:
        return ret, 0
    

def union_2dict(left_dict, right_dict, bound1, bound2, coord = 0):#right_dict = {classes : [np.array[boxes], np.array[confidance]]}
    if len(left_dict) == 0:
        return right_dict
    if len(right_dict) == 0:
        return left_dict
    
    classes = np.unique(list(left_dict.keys())+ list(right_dict.keys()))
    union_dict = {}
    for cls in classes:
        
        if cls not in left_dict:
            union_dict[cls] = right_dict[cls]
            continue
        if cls not in right_dict:
            union_dict[cls] = left_dict[cls]
            continue
            
        left_indexx = np.where(left_dict[cls][0][:,coord+2] < bound1)[0]
        if left_indexx.shape[0] != 0:
            union_dict[cls] = [left_dict[cls][0][left_indexx], left_dict[cls][1][left_indexx]]
            
        right_indexx = np.where(right_dict[cls][0][:,coord] > bound2)[0]
        if right_indexx.shape[0] != 0:
            if cls in union_dict:
                union_dict[cls][0] = np.vstack([union_dict[cls][0], right_dict[cls][0][right_indexx]])
                union_dict[cls][1] = np.append(union_dict[cls][1], right_dict[cls][1][right_indexx])
            else:
                union_dict[cls] = [right_dict[cls][0][right_indexx], right_dict[cls][1][right_indexx]]
        
        left_index = np.where(left_dict[cls][0][:,coord+2] >= bound1)[0]
        right_index = np.where(right_dict[cls][0][:,coord] <= bound2)[0]
        left_bboxes = [left_dict[cls][0][left_index], left_dict[cls][1][left_index]]
        right_bboxes = [right_dict[cls][0][right_index], right_dict[cls][1][right_index]]
        
        pair_index = np.full(right_bboxes[0].shape[0], -1, dtype=np.int32)
        
        i = 0
        for id_boxr in range(right_bboxes[0].shape[0]):
            boxr = right_bboxes[0][id_boxr]
            max_S = 0
            max_id = -1
            
            if coord:
                S_boxr = f_S(boxr[0], max(boxr[1], bound1) , boxr[2], min(boxr[3], bound2))
            else:
                S_boxr = f_S(max(boxr[0], bound1), boxr[1], min(boxr[2], bound2), boxr[3])   
                
            for id_boxl in range(left_bboxes[0].shape[0]):
                boxl = left_bboxes[0][id_boxl] 
                
                if coord:
                    S_boxl = f_S(boxl[0], max(boxl[1], bound1) , boxl[2], min(boxl[3], bound2))
                else:
                    S_boxl = f_S(max(boxl[0], bound1), boxl[1], min(boxl[2], bound2), boxl[3])
                 
                ret, S = union(boxr, boxl)
                S_min = min(S / S_boxr, S / S_boxl)
                S_max = max(S / S_boxr, S / S_boxl)
                
                if (S_min >= 0.7 or S_max >= 0.9) and (S_max > max_S):
                    max_S = S_max
                    max_id = id_boxl
            pair_index[id_boxr] = max_id
            
          
        for id_r, id_l in enumerate(pair_index):
            if id_l != -1:
                left_bboxes[0][id_l] = [   min(left_bboxes[0][id_l][0], right_bboxes[0][id_r][0]),
                                           min(left_bboxes[0][id_l][1], right_bboxes[0][id_r][1]),
                                           max(left_bboxes[0][id_l][2], right_bboxes[0][id_r][2]),
                                           max(left_bboxes[0][id_l][3], right_bboxes[0][id_r][3])  ]
                
                left_bboxes[1][id_l] = max(left_bboxes[1][id_l], right_bboxes[1][id_r])
            else:
                left_bboxes[0] = np.vstack([left_bboxes[0], right_bboxes[0][id_r]])
                left_bboxes[1] = np.append(left_bboxes[1], right_bboxes[1][id_r])
                
        if cls in union_dict:
            union_dict[cls][0] = np.vstack([union_dict[cls][0], left_bboxes[0]])
            
            union_dict[cls][1] = np.append(union_dict[cls][1], left_bboxes[1])
        else:
            union_dict[cls] = left_bboxes
            
    return union_dict
                
            
                                          
                                             
def detect_panorama(image, model, delta, side_x = 640, side_y = 640): #delta [0,1]
    
    delta_x = np.int32(side_x - delta * side_x)
    delta_y = np.int32(side_y - delta * side_y)
    
    h, w = image.shape[:2]
    n_wblocks = w // delta_x
    n_hblocks = h // delta_y
    
    all_bboxes = {}
    
    for y in range(0, n_hblocks):
        y_pix = y * delta_y
        koef = scaling_f(y)
        
        row_bboxes = {}
        
        for x in range(0, n_wblocks):
            
            x_pix = x * delta_x
            im_cropped = image[y_pix : y_pix+side_y, x_pix : x_pix+side_x]
            
            im_resized = cv.resize(im_cropped, None, fx=1/koef, fy=1/koef)
            
            results = model(im_resized, device="mps")
            
            result = results[0]
            
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            classes = np.array(result.boxes.cls.cpu(), dtype="int")
            conf = np.array(result.boxes.conf.cpu(), dtype="float")
            cls_names = result.names      
            bboxes = koef * bboxes
            bboxes += [x_pix, y_pix, x_pix, y_pix]
            bboxes = np.int32(bboxes)
            
            bboxes[np.where(bboxes[:, 2] > x_pix + side_x - 1)[0], 2] = x_pix + side_x - 1
            bboxes[np.where(bboxes[:, 3] > y_pix + side_y - 1)[0], 3] = y_pix + side_y - 1
            
            dict_bboxes = dict.fromkeys(np.unique(classes))
            for cls in dict_bboxes:
                indexx = np.where(classes == cls)[0]
                dict_bboxes[cls] = [bboxes[indexx]]
                dict_bboxes[cls].append(conf[indexx])
                
            if x == 0:
                row_bboxes = dict_bboxes
            else:
                row_bboxes = union_2dict(row_bboxes, dict_bboxes, x_pix, x_pix + side_x - delta_x, coord = 0)
                
        
        if y == 0:
            all_bboxes = row_bboxes
        else:
            all_bboxes = union_2dict(all_bboxes, row_bboxes, y_pix, y_pix + side_y - delta_y, coord = 1)
                            
    return all_bboxes, cls_names

def detect_blocks(image, model, delta, side_x = 640, side_y = 640): #delta [0,1]
    
    delta_x = np.int32(side_x - delta * side_x)
    delta_y = np.int32(side_y - delta * side_y)
    
    h, w = image.shape[:2]
    n_wblocks = w // delta_x
    n_hblocks = h // delta_y
    
    all_bboxes = []
    all_classes = []
    all_conf = []
    
    for y in range(0, n_hblocks):
        y_pix = y * delta_y
        koef = scaling_f(y)
        
        row_bboxes = {}
        
        for x in range(0, n_wblocks):
            
            x_pix = x * delta_x
            im_cropped = image[y_pix : y_pix+side_y, x_pix : x_pix+side_x]
          
            im_resized = cv.resize(im_cropped, None, fx=1/koef, fy=1/koef)
            
            results = model(im_resized, device="mps")
            
            result = results[0]
            
            bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
            classes = np.array(result.boxes.cls.cpu(), dtype="int")
            conf = np.array(result.boxes.conf.cpu(), dtype="float")
            cls_names = result.names
            
            bboxes = koef * bboxes
            bboxes += [x_pix, y_pix, x_pix, y_pix]
            bboxes = np.int32(bboxes)
            
            bboxes[np.where(bboxes[:, 2] > x_pix + side_x - 1)[0], 2] = x_pix + side_x - 1
            bboxes[np.where(bboxes[:, 3] > y_pix + side_y - 1)[0], 3] = y_pix + side_y - 1
            
            all_bboxes.append(bboxes)
            all_classes.append(classes)
            all_conf.append(conf)
    return all_bboxes, all_classes, all_conf, cls_names









    



    

        















