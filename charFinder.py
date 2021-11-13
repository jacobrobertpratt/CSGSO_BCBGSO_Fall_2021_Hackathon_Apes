from collections import Counter

import numpy as np
import cv2
import matplotlib.pyplot as plt

def charImageGetter(file_name):


    imageCv = cv2.imread(file_name)
    imageCv2 = cv2.imread(file_name)

    grayscale = cv2.cvtColor(imageCv, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(grayscale, 100, 255, cv2.THRESH_BINARY)
    imageCv2 = cv2.bitwise_and(grayscale, grayscale, mask=mask)
    ret, new_img = cv2.threshold(imageCv2, 100, 255, cv2.THRESH_BINARY)
    grayscale = cv2.cvtColor(imageCv, cv2.COLOR_BGR2GRAY)

    cv2.imshow('Please work', grayscale)


    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))

    dilated = cv2.dilate(new_img, kernel, iterations=1)
    smoothed = cv2.erode(dilated, kernel, iterations=3)

    contours, hierarchy = cv2.findContours(smoothed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imageCv, contours, -1, (0, 255, 0, 3))

    hArr = np.zeros(200)
    wArr = np.zeros(200)
    yArr = np.zeros(200)
    xArr = np.zeros(200)
    i = 0
    croppedImages = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        hArr[i] = h
        wArr[i] = w
        yArr[i] = y
        xArr[i] = x
        if w < 7 and h < 7 or w > 300 and h > 300:
            continue
        cv2.rectangle(imageCv, (x, y), (x + w, y + h), (255, 0, 255), 2)
        croppedImages.append(smoothed[y:y + h, x: x + w])
        i += 1

    unsorted = []
    imgsortedy = []
    for count, p in enumerate(croppedImages):
        tempArr = []
        tempArr.append(xArr[count])
        tempArr.append(xArr[count] + wArr[count])
        tempArr.append(yArr[count])
        tempArr.append(yArr[count]+hArr[count])
        tempArr.append(p)
        unsorted.append(tempArr)

    q = 0
    while len(unsorted) > 0:
        image = unsorted.pop()
        fits = False
        for i in imgsortedy:
            if i[0] > image[0] and i[1] > image[1]:
                i[2].append(image)
                fits = True
                break
            elif i[0] < image[0] and i[1] > image[1]:
                i[0] = image[0]
                i[2].append(image)
                fits = True
                break
            elif i[0] > image[0] and i[1] < image[1]:
                i[1] = image[1]
                i[2].append(image)
                fits = True
                break
        if fits == False:
            imgsortedy.append([image[0], image[1], [image]])

    truimgsortedy = []



    while len(imgsortedy) > 0:
        tempMin = imgsortedy[0][0]
        tempIndex = 0
        for count,i in enumerate(imgsortedy):
            if i[0]<tempMin:
                tempMin = i[0]
                tempIndex = count
        truimgsortedy.append(imgsortedy.pop(tempIndex))


    finalArray = []

    for i in truimgsortedy:
        tempList = i[2]
        while len(tempList) > 0:
            count = 0
            tempMin = tempList[0][2]
            tempLoc = 0
            for z in tempList:
                if z[2] < tempMin:
                    tempMin = z[2]
                    tempLoc = count
                count += 1
            finalArray.append(tempList.pop(tempLoc)[4])

    j = 0
    for character in croppedImages:
        croppedImages[j] = 1-(cv2.resize(finalArray[j], (28, 28), interpolation=cv2.INTER_CUBIC)/255.)
        j += 1

    return croppedImages

myList = charImageGetter('officalTestImage.jpg')


plt.show()

cv2.waitKey(0)
