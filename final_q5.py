# coding=utf-8
#!/usr/bin/python
'''
Created on 2017��9��16��
 
@author: zhangjian
'''
import cv2  
import numpy as np  
def get_mode(arr):  
    mode = [];  
    arr_appear = dict((a, arr.count(a)) for a in arr);    
    if max(arr_appear.values()) == 1:  
        return;  
    else:  
        for k, v in arr_appear.items():  
            if v == max(arr_appear.values()):  
                mode.append(k);  
    return mode;  
def translate(image, x, y):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    return shifted   
def kuo(iamge,x,y): 
    rows1 = iamge.shape[0]
    cols1 = iamge.shape[1]
    out = np.zeros((rows1+x,cols1+y,3), dtype='uint8')
    out[x:x+rows1,y:y+cols1] = np.dstack([iamge])
    return out
def pingjie1(img1,img2): 
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = np.zeros((max([rows1,rows2]),cols2,3), dtype='uint8')
    out[:rows2-4,:cols2] = np.dstack([img2[4:rows2,:cols2]])
    out[:rows1,:cols1] = np.dstack([img1]) 
    return out   
def pingjie(img1,img2): 
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
    out[:rows1,:cols1] = np.dstack([img1[:rows1,:cols1]])
    out[rows1-264:rows1-264+rows2,cols1-300:cols1-300+cols2] = np.dstack([img2])    
    return out
def pingjie2(img1,img2,img3): 
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    rows3 = img3.shape[0]
    cols3 = img3.shape[1]
    out = np.zeros((rows3+rows1,cols1+cols2,3), dtype='uint8')   
    out[:rows1,:cols1] = np.dstack([img1[:rows1,:cols1]])
    out[rows1-264:rows1-264+rows3-4,cols1-300:cols1-300+cols3] = np.dstack([img3[4:rows3,:cols3]])
    out[rows1-264:rows1-264+rows2,cols1-300:cols1-300+cols2] = np.dstack([img2]) 
    return out
def drawMatches(img1, kp1, img2, kp2, matches):
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
 
    out = np.zeros((max([rows1,rows2]),cols1+cols2,3), dtype='uint8')
 
    out[:rows1,:cols1] = np.dstack([img1])
 
    out[:rows2,cols1:] = np.dstack([img2])
    a=[]
    b=[]
    c=[]
    d=[]
    for mat in matches:
 
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx
 
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt
        a.append([x1,y1])
        b.append([x2,y2])
        c.append(x1-x2)
        d.append(y1-y2)

        cv2.circle(out, (int(x1),int(y1)), 4, (0, 0, 255), 1)   
        cv2.circle(out, (int(x2)+cols1,int(y2)), 4, (0, 0, 255), 1)

        cv2.line(out, (int(x1),int(y1)), (int(x2)+cols1,int(y2)), (0, 0, 255), 1)
    return out,a,b,c,d
img1 = cv2.imread("./zuo.jpg")  
img2 = cv2.imread("./zhong.jpg")        
img1=cv2.resize(img1,(300,200))  
img2=cv2.resize(img2,(300,200)) 
img2= kuo(img2,140,300)
orb = cv2.ORB()     
kp1, des1 = orb.detectAndCompute(img1,None)  
kp2, des2 = orb.detectAndCompute(img2,None)  
bf = cv2.BFMatcher(cv2.NORM_HAMMING)  
matches = bf.knnMatch(des1, trainDescriptors = des2, k = 2)  
good = [m for (m,n) in matches if m.distance < 0.6*n.distance]  
img3,a,b,c,d = drawMatches(img1,kp1,img2,kp2,good[0:40])
h, status = cv2.findHomography(np.matrix(np.float32(a)),np.matrix(np.float32(b)),cv2.RANSAC,5.0)
im_out = cv2.warpPerspective(img1, h,(600, 400))
img1 = cv2.imread("./zhong.jpg")  
img2 = cv2.imread("./you.jpg")    
img1=cv2.resize(img1,(300,220))  
img2=cv2.resize(img2,(300,220)) 
orb = cv2.ORB()     
kp1, des1 = orb.detectAndCompute(img1,None)  
kp2, des2 = orb.detectAndCompute(img2,None)  
bf = cv2.BFMatcher(cv2.NORM_HAMMING)  
matches = bf.knnMatch(des1, trainDescriptors = des2, k = 2)  
good = [m for (m,n) in matches if m.distance < 0.6*n.distance]  
img3,a,b,c,d = drawMatches(img1,kp1,img2,kp2,good[0:40])
h, status = cv2.findHomography(np.matrix(np.float32(b)),np.matrix(np.float32(a)),cv2.RANSAC,5.0)
im_out_1 = cv2.warpPerspective(img2, h,(600, 300))
img1 = cv2.imread("./zhong.jpg") 
img1=cv2.resize(img1,(300,210)) 
print im_out.shape[0]
print im_out.shape[1]
print img1.shape[0]
print img1.shape[1]
print im_out_1.shape[0]
print im_out_1.shape[1]

img31_out=pingjie2(im_out,img1,im_out_1)
cv2.imshow('image',img31_out)
cv2.waitKey(0)   
cv2.destroyAllWindows()

