
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 16:22:26 2019

@author: user
"""

farhan
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt



list = os.listdir("C:/Users/Care/Desktop/iris segmentation/img") # dir is your directory path
number_files = len(list)
print (number_files)
print("database connected")
 #w=10
#h=10
#fig=plt.figure(figsize=(8, 8))
#columns = 6
#rows = 5
#for i in range(1, columns*rows +1):
    #img=cv2.imread("./img/.bmp")
#    img = np.random.randint(10, size=(h,w))
   # fig.add_subplot(rows, columns, i)
    #plt.imshow(img)
plt.show()
img = cv2.imread("./img/1.bmp")

#output=img.copy()

# show image format 
print(img)

# convert image to RGB color for matplotlib
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# show image with matplotlib
plt.imshow(img)

# convert image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.show()

# grayscale image represented as an array
print(gray_img)

plt.imshow(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2RGB))
cv2.imwrite("./output/gray_img.png", gray_img)

plt.show()

#threshold on blurred image
gray_blur_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshold_img_blur = cv2.threshold(gray_blur_img, 100, 255, cv2.THRESH_BINARY)
cv2.imwrite('output/oy-gaussian-blur-5-thresh.jpg', threshold_img_blur)
plt.imshow(cv2.cvtColor(threshold_img_blur, cv2.COLOR_GRAY2RGB))
plt.show()



# using adaptive threshold instead of global
adaptive_thresh = cv2.adaptiveThreshold(gray_img,255,\
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                                         cv2.THRESH_BINARY,11,2)
plt.imshow(cv2.cvtColor(adaptive_thresh, cv2.COLOR_GRAY2RGB))
cv2.imwrite("./output/adaptive_thresh.png", adaptive_thresh)
plt.show()

_, eye_binary = cv2.threshold(gray_img, 130, 255, cv2.THRESH_BINARY)

# invert image to get eye
eye_binary = cv2.bitwise_not(eye_binary)
plt.imshow(cv2.cvtColor(eye_binary, cv2.COLOR_GRAY2RGB))
cv2.imwrite("./output/eye.png", eye_binary)


img = cv2.imread("./output/gray_img.png",0)
img = cv2.medianBlur(img,5)
cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,
                            param1=80,param2=50,minRadius=0,maxRadius=0)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

plt.title("Detected Iris")
plt.imshow(cimg)
plt.show()
img = cv2.imread("./img/iris_segmented.jpg")
plt.imshow(img)
plt.show()

