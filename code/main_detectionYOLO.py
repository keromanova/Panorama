import detectionYOLO as dyolo
import cv2 as cv
from ultralytics import YOLO
import numpy as np
import torch
import glob
import os


path = "../SVO/day/firstrow/image%06d.jpg"
path_results = "../results_images"
number_image = 10

model = YOLO('yolov8n.pt')


frame = cv.imread(path % number_image)
delta = 0.3
all_bboxes, all_classes, all_conf, cls_names = dyolo.detect_blocks(frame, model, delta, side_x = 1100, side_y = 800)

for bboxes, classes, confs in zip(all_bboxes, all_classes, all_conf):
    for bbox, cls, conf in zip(bboxes, classes, confs):
        (x1, y1, x2, y2) = bbox
        color = (0, 0, 255)
        cv.rectangle(frame, (x1, y1), (x2, y2), color, 4)
        cv.putText(frame, cls_names[cls]+": %0.3f"%conf, (x1, y1-2), cv.FONT_HERSHEY_PLAIN, 4, color, 2)
        
cv.imshow("Img", frame)
cv.waitKey(0)
cv.imwrite(path_results + "/panorama-detect-blocks.jpg", frame)



frame = cv.imread(path % number_image)
delta = 0.3
all_bboxes, cls_names = dyolo.detect_panorama(frame, model, delta, side_x = 1100, side_y = 800)


for cls in all_bboxes:
    bboxes = all_bboxes[cls][0]
    
    for i, bbox in enumerate(bboxes):
        [x1, y1, x2, y2] = bbox
        cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
        cv.putText(frame, cls_names[cls]+": %0.3f"%all_bboxes[cls][1][i], (x1, y1-2), cv.FONT_HERSHEY_PLAIN, 4,
                   (0, 0, 255), 2)


cv.imshow("Img", frame)
cv.waitKey(0)
cv.imwrite(path_results + "/panorama-detect-union.jpg", frame)




n_images = 250
path = "../SVO/day/firstrow/image%06d.jpg"
path_res = "../results_images/detect_blocks"


os.mkdir(path_res)



for idim in range(1, n_images+1):
    frame = cv.imread(path % idim)
    delta = 0.3
    all_bboxes, all_classes, all_conf, cls_names = dyolo.detect_blocks(frame, model, delta, side_x = 1100, side_y = 800)
    for bboxes, classes, confs in zip(all_bboxes, all_classes, all_conf):
        for bbox, cls, conf in zip(bboxes, classes, confs):
            (x1, y1, x2, y2) = bbox
            color = (0, 0, 0)
            cv.rectangle(frame, (x1, y1), (x2, y2), color, 4)
            cv.putText(frame, cls_names[cls]+": %0.3f"%conf, (x1, y1-2), cv.FONT_HERSHEY_PLAIN, 4, color, 2)
            
    cv.imwrite(path_res + "/image%06d.jpg" % idim, frame[:-1, :-1, :])


cmd = 'ffmpeg -f image2 -i '+ path_res + "/image%06d.jpg"  + " " + path_res + "/../detect_blocks.mp4"
os.system(cmd)





n_images = 250
path = "../SVO/day/firstrow/image%06d.jpg"
path_res = "../results_images/detect_panorama"


os.mkdir(path_res)



for idim in range(1, n_images+1):
    frame = cv.imread(path % idim)
    delta = 0.3
    all_bboxes, cls_names = dyolo.detect_panorama(frame, model, delta, side_x = 1100, side_y = 800)
    for cls in all_bboxes:
        bboxes = all_bboxes[cls][0]

        for i, bbox in enumerate(bboxes):
            [x1, y1, x2, y2] = bbox
            color = (0, 0, 0)
            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv.putText(frame, cls_names[cls]+": %03f"%all_bboxes[cls][1][i], (x1, y1-2), cv.FONT_HERSHEY_PLAIN, 4, color, 2)
            
    cv.imwrite(path_res + "/image%06d.jpg" % idim, frame[:-1, :-1, :])

cmd = 'ffmpeg -f image2 -i '+ path_res + "/image%06d.jpg"  + " " + path_res + "/../detect_panorama.mp4"
os.system(cmd)

cv.destroyAllWindows()