import numpy as np 
import cv2 
import dlib 
import sys
import math 
import skimage.io


def face_euler_pose(img):
   
    #获取相机矩阵
    focal_length = img.shape[1] 
    center = (img.shape[1]/2, img.shape[0]/2) 
    camera_matrix = np.array( [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype = "double" ) 
    # opencv内相机光学畸变置零
    dist_coeffs = np.zeros((4,1)) 
    #相机姿态估计，内含旋转矩阵/平移矩阵
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,flags=cv2.SOLVEPNP_ITERATIVE)
    #偏转角
    theta = cv2.norm(rotation_vector, cv2.NORM_L2) 
    # 转化为欧拉四元向量
    w = math.cos(theta / 2) 
    x = math.sin(theta / 2)*rotation_vector[0][0] / theta 
    y = math.sin(theta / 2)*rotation_vector[1][0] / theta 
    z = math.sin(theta / 2)*rotation_vector[2][0] / theta 
    ysqr = y * y 
    # 偏向角 
    t0 = 2.0 * (w * x + y * z) 
    t1 = 1.0 - 2.0 * (x * x + ysqr) 
    print('t0:{}, t1:{}'.format(t0, t1)) 
    pitch = math.atan2(t0, t1) 
    # 俯仰角 
    t2 = 2.0 * (w * y - z * x) 
    if t2 > 1.0: 
        t2 = 1.0 
    if t2 < -1.0: 
        t2 = -1.0 
    yaw = math.asin(t2) 
    # 横滚角 
    t3 = 2.0 * (w * z + x * y) 
    t4 = 1.0 - 2.0 * (ysqr + z * z) 
    roll = math.atan2(t3, t4) 
     
    # 单位转换：将弧度转换为度 
    Y = abs(int((pitch/math.pi)*180)) -180
    X = int((yaw/math.pi)*180) 
    Z = int((roll/math.pi)*180)
    #print('俯仰角:{}, 偏向角:{}, 横滚角:{}'.format(pitch, yaw, roll))
    return Y, X, Z
       

load_name = sys.argv[1]
img = skimage.io.imread(load_name)

'''
#旋转测试
rows, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 40, 0.5)
img = cv2.warpAffine(img, M, (cols, rows))
'''

# 预加载Dlib数据
detector = dlib.get_frontal_face_detector() 
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
dets = detector( img, 0 ) 
landmark = predictor(img, dets[0]) 


#人脸特征点矩阵
image_points = np.array([ (landmark.part(30).x, landmark.part(30).y), # 鼻尖
                          #(landmark.part(8).x, landmark.part(8).y), # 下巴 
                          (landmark.part(36).x, landmark.part(36).y), # 左眼
                          (landmark.part(45).x, landmark.part(45).y), # 右眼
                          (landmark.part(48).x, landmark.part(48).y), # 左嘴角
                          (landmark.part(54).x, landmark.part(54).y) # 右嘴角
                           ], dtype="double")

#图像基准点矩阵
model_points = np.array([ (0.0, 0.0, 0.0), # 鼻尖
                          #(0.0, -330.0, -65.0), # 下巴 
                          (-225.0, 170.0, -135.0), # 左眼 
                          (225.0, 170.0, -135.0), # 右眼
                          (-150.0, -150.0, -125.0), # 左嘴角 
                          (150.0, -150.0, -125.0) # 右嘴角 
                        ]) 


pitch, yaw, roll = face_euler_pose(img)
eulerangle = 'Y:{}, X:{}, Z:{}'.format(pitch, yaw, roll)

#偏转法线
#(nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs) 
#p1 = ( int(image_points[0][0]), int(image_points[0][1])) 
#p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1])) 
#cv2.line(img, p1, p2, (255,0,0), 2)

for p in image_points:
    cv2.circle(img, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
cv2.putText( img, eulerangle, (0, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 1 ) 
cv2.imshow("Output", img) 
cv2.waitKey(0)




