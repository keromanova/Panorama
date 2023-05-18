import os
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob



def from_video_to_images(path_to_video, start_time, time, to_dir):
    cmd = 'ffmpeg -i ' + path_to_video +' -ss ' + start_time + ' -t ' + time + ' ' + to_dir + '/image%d.jpg'
    os.system(cmd)


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

    
def resized_image(cut_coord, coef_x, coef_y, path, cam_coord, cam, num_image):
    cut_x1, cut_x2, cut_y1, cut_y2 = cut_coord
    cam_x = cam_coord[cam, 0]
    cam_y = cam_coord[cam, 1]
    
    image = cv.imread(path%(cam + 1, num_image))[cut_y1 - cam_y:cut_y2 - cam_y, cut_x1 -cam_x:cut_x2 - cam_x, :]
    
    dim = (int(image.shape[1] * coef_x), int(image.shape[0] * coef_y))
    if dim[0]*dim[1] == 0:
                return False, []
    resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
    return True, resized



def make_display(area_x1, area_y1, area_x2, area_y2, screen_width, screen_hight, cam_coord, num_image, path):
    display = np.full((screen_hight, screen_width, 3), 0,  dtype=np.uint8)
    
    coef_x = screen_width / (area_x2 - area_x1)
    coef_y = screen_hight / (area_y2 - area_y1)

    for cam in range(cam_coord.shape[0]):
        cam_x = cam_coord[cam, 0]
        cam_y = cam_coord[cam, 1]
        
        ret, cut_coord = cut_area(cam_x, cam_y, cam_coord[cam, 2], cam_coord[cam, 3], area_x1, area_y1, area_x2, area_y2)
        if ret:
            cut_x1, cut_x2, cut_y1, cut_y2 = cut_coord
            image = cv.imread(path%(cam + 1, num_image))[cut_y1 - cam_y:cut_y2 - cam_y, cut_x1 - cam_x:cut_x2 - cam_x]
            
            
            dim = (int(image.shape[1] * coef_x), int(image.shape[0] * coef_y))
            if dim[0]*dim[1] == 0:
                continue
            
            resized = cv.resize(image, dim, interpolation = cv.INTER_AREA)
            
            
            start_y = int((cut_y1 - area_y1) * coef_y)
            start_x = int((cut_x1 - area_x1) * coef_x)
            display[start_y:start_y + dim[1], start_x:start_x+dim[0], :] = resized
    
    return display

def make_display_(area_x1, area_y1, screen_width, screen_hight, cam_coord, num_image, path):

    display = np.full((screen_hight, screen_width, 3), 0,  dtype=np.uint8)
    
    area_x2 = area_x1 + screen_width
    area_y2 = area_y1 + screen_hight

    for cam in range(cam_coord.shape[0]):#[1, 2, 3]:
        cam_x = cam_coord[cam, 0]
        cam_y = cam_coord[cam, 1]
        ret, cut_coord = cut_area(cam_x, cam_y, cam_coord[cam, 2], cam_coord[cam, 3], area_x1, area_y1, area_x2, area_y2)
        if ret:
            cut_x1, cut_x2, cut_y1, cut_y2 = cut_coord
            image = cv.imread(path%(cam + 1, num_image))[cut_y1 - cam_y:cut_y2 - cam_y, cut_x1 - cam_x:cut_x2 - cam_x]
            display[cut_y1 - area_y1:cut_y2 - area_y1, cut_x1 - area_x1:cut_x2 - area_x1, :] = image
    
    return display[:, :, :]

def make_display_frames(scr_left_up, scr_right_down, screen_width, screen_hight, cam_coord, start_image, number_image, path_data, path_res):
    os.mkdir(path_res)
    for count in range(number_image):
        area_x1, area_y1 = scr_left_up[count]
        area_x2, area_y2 = scr_right_down[count]
        
        display = make_display(area_x1, area_y1, area_x2, area_y2, screen_width, screen_hight, cam_coord, start_image + count, path_data)
        cv.imwrite(path_res+'/image%d.jpg'%(count), display)
    cmd = 'ffmpeg -f image2 -i '+ path_res+'/image%d.jpg ' + path_res+'/video.mp4'
    os.system(cmd)
    
def im_show(im, gray = None):
    fig, ax = plt.subplots()
    ax.imshow(im, cmap = gray)
    fig.set_figwidth(43)
    fig.set_figheight(27)
    plt.show()

def im_show2(im1, im2, gray = None):
    im = cv.hconcat([im1, im2])
    fig, ax = plt.subplots()
    ax.imshow(im, cmap=gray)
    fig.set_figwidth(43)
    fig.set_figheight(27)
    plt.show()

def homografy(im1, im2):
    
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im1, None)
    kp2, des2 = sift.detectAndCompute(im2, None)

    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)

    pts1 = []
    pts2 = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
            
    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    if len(pts1) != 0:
        F, mask = cv.findFundamentalMat(pts1, pts2,  cv.FM_RANSAC)
        if type(mask) != type(None):
            pts1 = pts1[mask.ravel()==1]
            pts2 = pts2[mask.ravel()==1]
        else:
            return False, []
    else:
        return False, []
    H, _ = cv.estimateAffinePartial2D(pts2, pts1, False, method=cv.RANSAC)
    return H

def rotate(im, H):
    #0-----1
    #|     |
    #|     |
    #3-----2
    G = H.copy()
    R = G[:,:2]

    hight = im.shape[0]
    width = im.shape[1]
    points = np.zeros((4, 2), dtype=np.float32)
    points[0] = [0., 0.]
    points[1] = np.dot(R, [width - 1, 0])
    points[2] = np.dot(R, [width - 1, hight - 1])
    points[3] = np.dot(R, [0, hight - 1])

    dx = np.min(points[:,0])
    dy = np.min(points[:,1])
    points[:,0] -= dx
    points[:,1] -= dy

    size = (np.int32(np.max(points[:,0])), np.int32(np.max(points[:,1])))
    G[:,2] = [-dx, -dy]
    return G, size

def rotate_and_cut(im, H):
    #0-----1
    #|     |
    #|     |
    #3-----2
    G = H.copy()
    R = G[:,:2]

    hight = im.shape[0]
    width = im.shape[1]
    points = np.zeros((4, 2), dtype=np.float32)
    points[0] = [0., 0.]
    points[1] = np.dot(R, [width - 1, 0])
    points[2] = np.dot(R, [width - 1, hight - 1])
    points[3] = np.dot(R, [0, hight - 1])

    y1 = max(points[0][1], points[1][1])
    y2 = min(points[2][1], points[3][1])
    
    x1 = max(points[0][0], points[3][0])
    x2 = min(points[1][0], points[2][0])
    #print(x1, x2, y1, y2)
    
    
    dy = min(points[0][1] - points[1][1], 0.)
    dx = min(points[0][0] - points[3][0], 0.)
    #print(dx, dy)

    size = (np.int32(x2 - x1 + 1), np.int32(y2 - y1 + 1))
     
    G[:,2] = [dx, dy]
    return G, size

def stich_images(cam_coord, images, defolt_color = 0):
    
    screen_width = np.max(cam_coord[:,0] + cam_coord[:,2])
    screen_hight = np.max(cam_coord[:, 1] + cam_coord[:,3])
    
    dim = (screen_hight, screen_width, 3)
    if np.size(images[0].shape) == 2:
        dim = (screen_hight, screen_width)
    display = np.full(dim, defolt_color,  dtype=np.uint8)
    
    for im_id in range(len(images)):
        cam_x = cam_coord[im_id][0]
        cam_y = cam_coord[im_id][1]
        im = images[im_id]
        d = display[cam_y:cam_y + im.shape[0], cam_x:cam_x + im.shape[1]]
        display[cam_y:cam_y + im.shape[0], cam_x:cam_x + im.shape[1]] = np.where(im == defolt_color, d, im)
  
    return display

def camera_params(chessboard, path, typeim):
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objpoints = []
    imgpoints = [] 
    objp = np.zeros((1, chessboard[0] * chessboard[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:chessboard[0], 0:chessboard[1]].T.reshape(-1, 2)
    prev_img_shape = None
    images = glob.glob(path+'/*.'+typeim)
    for fname in images:
        img = cv.imread(fname)
        if img.shape[0] > img.shape[1]:
            img = cv.rotate(img,cv.ROTATE_90_COUNTERCLOCKWISE)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)

        ret, corners = cv.findChessboardCorners(gray, chessboard, cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE)
    
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray, corners, (11,11),(-1,-1), criteria)
        
            imgpoints.append(corners2)
    
    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h,  w = gray.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    return [mtx, dist, newcameramtx, roi]

def undistort_image(im, cam_params):
    mtx, dist, newcameramtx, roi = cam_params
    x, y, w, h = roi
    return cv.undistort(im, mtx, dist, None, newcameramtx)[y:y+h, x:x+w]



def integral(im):
    values, counts = np.unique(im, return_counts=True)
    return np.sum(values * counts)

def correction_f(poly):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(np.round(np.polyval(poly, i)), 0, 255)
    return lookUpTable

def correction_id():
    return correction_f([1., 0.])

def im_show2(im1, im2, gray = None):
    im = cv.hconcat([im1, im2])
    fig, ax = plt.subplots()
    ax.imshow(im, cmap=gray)
    fig.set_figwidth(43)
    fig.set_figheight(27)
    plt.show()
    
def find_correction(Y1, Y2):
    center = [127., 127.]
    x = [0., center[0], 255.]
    y = [0., center[1], 255.]

    error = integral(Y1) - integral(Y2)
    sign = np.sign(error)
    func = correction_id()
    y_list = [y[1] + sign * i for i in range(1, 100)]
    
    for y1 in y_list:
        y[1] = y1
        poly = np.polyfit(x, y, 2)
        new_func = correction_f(poly)
        new_Y2 = cv.LUT(Y2, new_func)
        new_error = integral(Y1) - integral(new_Y2)
        if np.sign(new_error) != sign:
            #print(y[1])
            if abs(new_error) > abs(error):
                #print('e')
                return func
            return new_func
        error = new_error
        func = new_func
    return func

def ret_functions(im1, im2, balanced_points1, balanced_points2, d = 25):
    functions = []

    p = 0
    point = balanced_points1[p]
    Y1 = im1[point[1]:point[1] + d, point[0]:point[0] + d]
    point = balanced_points2[p]
    Y2 = im2[point[1]:point[1] + d, point[0]:point[0] + d]
    
    functions.append(find_correction(Y1, Y2)[0])

    p = 1
    point = balanced_points1[p]
    Y1 = im1[point[1] - d//2:point[1] + d // 2 + 1, point[0]:point[0] + d]
    point = balanced_points2[p]
    Y2 = im2[point[1] - d//2:point[1] + d // 2 + 1, point[0]:point[0] + d]
    
    functions.append(find_correction(Y1, Y2)[0])

    p = 2
    point = balanced_points1[p]
    Y1 = im1[point[1] - d + 1:point[1] + 1, point[0]:point[0] + d]
    point = balanced_points2[p]
    Y2 = im2[point[1] - d + 1:point[1] + 1, point[0]:point[0] + d]
    
    functions.append(find_correction(Y1, Y2)[0])


    functions.append(functions[0])
    functions.append(functions[2])

    functions.append(functions[0])
    functions.append(functions[1])
    functions.append(functions[2])
    
    functions = np.array(functions)
    return functions

def distance(point1, point2):
    return np.sqrt(np.power(point1[0] - point2[0], 2) + np.power(point1[1] - point2[1], 2))

def weights(distances):
    d = np.min(distances) + np.max(distances)
    weight = d - distances
    return weight /  (d * distances.shape[0] - np.sum(distances))

def ret_weights(point, points2):
    distances = np.zeros(points2.shape[0], dtype=np.int32)
    for i in range(distances.shape[0]):
        distances[i] = distance(point, points2[i])
    return weights(distances)

def return_W(cam1, cam2, cam_coord):
    res, area = cut_area(cam_coord[cam1][0], cam_coord[cam1][1], 
                   cam_coord[cam1][0] + cam_coord[cam1][2], cam_coord[cam1][1] + cam_coord[cam1][3],
                   cam_coord[cam2][0], cam_coord[cam2][1], 
                   cam_coord[cam2][0] + cam_coord[cam2][2], cam_coord[cam2][1] + cam_coord[cam2][3],)

    balanced_points2 = np.array([[area[0] - cam_coord[cam2][0], area[2] - cam_coord[cam2][1]], 
                                      [area[0] - cam_coord[cam2][0], (area[2] + area[3]) // 2 - cam_coord[cam2][1]], 
                                      [area[0] - cam_coord[cam2][0], area[3] - 1 - cam_coord[cam2][1] ]])

    points2 = np.zeros((8, 2), dtype=np.int32)
    points2[:3, :] = balanced_points2
    points2[3:5, :] =( balanced_points2 + [cam_coord[cam2][2]//2, 0])[::2]
    points2[5:8, :] = balanced_points2 + [cam_coord[cam2][2] - 1, 0]

    W = np.zeros((cam_coord[cam2][3], cam_coord[cam2][2], 8), dtype = np.float32)
    for y in range(W.shape[0]):
        for x in range(W.shape[1]):
            W[y, x, :] = ret_weights([x, y], points2)
    return W

def brightness_correction(im1, im2, cam1, cam2, cam_coord, W_list):
    res, area = cut_area(cam_coord[cam1][0], cam_coord[cam1][1], 
                   cam_coord[cam1][0] + cam_coord[cam1][2], cam_coord[cam1][1] + cam_coord[cam1][3],
                   cam_coord[cam2][0], cam_coord[cam2][1], 
                   cam_coord[cam2][0] + cam_coord[cam2][2], cam_coord[cam2][1] + cam_coord[cam2][3],)

    balanced_points1 = np.array([[area[0] - cam_coord[cam1][0], area[2] - cam_coord[cam1][1]], 
                                  [area[0] - cam_coord[cam1][0], (area[2] + area[3]) // 2 - cam_coord[cam1][1]], 
                                  [area[0] - cam_coord[cam1][0], area[3] - 1 - cam_coord[cam2][1]]])

    balanced_points2 = np.array([[area[0] - cam_coord[cam2][0], area[2] - cam_coord[cam2][1]], 
                                  [area[0] - cam_coord[cam2][0], (area[2] + area[3]) // 2 - cam_coord[cam2][1]], 
                                  [area[0] - cam_coord[cam2][0], area[3] - 1 - cam_coord[cam2][1] ]])
    
    functions = ret_functions(im1, im2, balanced_points1, balanced_points2, d = 25)

    correction_im2 = np.zeros(im2.shape, dtype = np.uint8)

    for y in range(correction_im2.shape[0]):
        for x in range(correction_im2.shape[1]):
            pix = im2[y, x]
            correction_im2[y,x] = np.clip(np.round(np.sum(W_list[cam2][y, x] * functions[:,pix])), 0, 255)
    return correction_im2  