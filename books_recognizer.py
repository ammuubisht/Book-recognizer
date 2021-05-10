# Importing Required Libraries
import cv2 as cv
import numpy as np
import os

orb = cv.ORB_create(nfeatures=1000)
path = 'ImageData'


# Importing and Reading Images 
readImages = []
classNames = []
rawImages = os.listdir(path)
for img in rawImages:
    currentImg = cv.imread(f'{path}/{img}', 0)
    readImages.append(currentImg)
    classNames.append(os.path.splitext(img)[0])



# Finding Keypoints, Descriptors of Query Images using ORB 
def find_des(readImages):
    des_list = []
    for imgs in readImages:
        kp, des = orb.detectAndCompute(imgs, None)
        des_list.append(des)
    return des_list

des_list = find_des(readImages)
print(f'Number of Descriptors: {des_list}')



''' Finding Descriptors in real-time captures using ORB and listing out 
    the good matches using BruteForce '''
    
def find_matches(img, desList, threshold=15):  
    # Set the threshold value as the minimum good matches req. to accurately predict the book
    
    kp2, des2 = orb.detectAndCompute(img,None)
    bf = cv.BFMatcher()
    matchesList = []
    final_value = -1
    try: 
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m,n in matches:
                if m.distance <0.75*n.distance:
                    good.append([m])
            matchesList.append(len(good))
    except:
        pass
    if len(matchesList)!=0:
        if max(matchesList) > threshold:
            final_value = matchesList.index(max(matchesList))
    return final_value



# Classifying Books in real-time through webcam
video = cv.VideoCapture(0)

while True:
    ret, img = video.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    match = find_matches(gray, des_list)
    if match != -1:
        cv.putText(img, classNames[match], (50,50), cv.FONT_HERSHEY_COMPLEX, 1.2, (0,255,0), 2)

    cv.imshow('Book Recognizer', img)
    cv.waitKey(1)