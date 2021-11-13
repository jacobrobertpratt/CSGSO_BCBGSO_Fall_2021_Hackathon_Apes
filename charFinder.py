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

    return croppedImages

myList = charImageGetter('officalTestImage.jpg')

cv2.imshow('ligma', myList[17])
plt.show()

cv2.waitKey()
