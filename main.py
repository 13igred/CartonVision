import cv2
import numpy
import re

import numpy as np
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\TesseractOCR\tesseract.exe'


# Globals
X = 0
Y = 0
W = 0
H = 0

# Find any rectangles as the best before is often written in these
# based on rectangle the RoI has been found
# these sizes will likely need some adjustment
def FindRectangle(img):

    # Render edges
    height, width = img.shape[:2]
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(imgGrey, (5, 5), 0)
    edged = cv2.Canny(blur, 140, 230, 3)  # Find edges

    cnts, hierarchies = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours


    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.015 * peri, True)

        if len(approx) == 4:
            X, Y, W, H = cv2.boundingRect(approx)

            # Adjust based on image size
            if W > 400 and 500 > H > 100:
                cv2.rectangle(img, (X, Y), (X + W, Y + H), (0, 255, 0), 1)
            else:
                X = 0
                Y = 0
                H = 0
                W = 0

    return X, Y, W, H


# Take a x position
# project out a rectangle
# capture how many contours are in that area
# if its over a certain threshold RoI has been found
def NoRectangleFound(img):
    X = 0
    Y = 0
    H = 0
    W = 0

    # Render edges
    height, width = img.shape[:2]
    imgGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(imgGrey, (5, 5), 0)
    # cv2.namedWindow(windowName, 3)  # unclear
    edged = cv2.Canny(blur, 140, 230, 3)  # Find edges

    cnts, hierarchies = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # Find contours

    ptX = []
    ptY = []
    # Find moments to calculate average position for a contour
    for c in cnts:
        M = cv2.moments(c)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            # cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
            # cv2.circle(img, (cx, cy), 7, (0, 0, 255), -1)
            ptX.append(cx)
            ptY.append(cy)
    RoIx = []
    RoIy = []
    index = 0
    for i in range(0, len(ptX)):
        # cv2.rectangle(img, (ptX[i] - 10, ptY[i] - 20), (ptX[i] + 75, ptY[i] + 20), (255, 0, 0), 1)
        count = 0
        # Project RoI
        startX = ptX[i] - 10
        startY = ptY[i] - 20
        w = ptX[i] + 75
        h = ptY[i] + 20
        # Assess region
        for j in range(0, len(ptX)):
            if startX < ptX[j] < w and startY < ptY[j] < h:
                count = count + 1
            # If count is over a threshold RoI has been found
            if count > 10:
                #cv2.rectangle(img, (ptX[i] - 10, ptY[i] - 20), (ptX[i] + 75, ptY[i] + 20), (255, 0, 0), 1)
                RoIx.append(ptX[i])
                RoIy.append(ptY[i])
                break
    # RoI - Rectangle Definition
    if len(RoIx) > 0 and len(RoIy) > 0:
        X = int(min(RoIx)) - 75
        Y = int(min(RoIy)) - 50
        W = int(max(RoIx)) + 50
        H = int(max(RoIy)) + 50 - Y
        # Draw RoI
        cv2.rectangle(img, (X, Y), (X + W, Y + H), (255, 0, 0), 1)
    return X, Y, W, H

# Do more image manipulation for tesseract
def FindText(img, threshLow, threshHigh, thickness):

    CIGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    CIBlur = cv2.blur(CIGray, (5, 5))
    CICanny = cv2.Canny(CIBlur, threshLow, threshHigh, 3)
    # CICanny = cv2.Canny(CIBlur, 45, 135, 3)
    # CICanny = cv2.Canny(CIBlur, 50, 150, 3)
    cnts, h = cv2.findContours(CICanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(CIGray, cnts, -1, (0, 255, 0), 1)
    CIErode = cv2.erode(CIGray, np.ones((5, 5)), iterations=1)
    text = pytesseract.image_to_string(CIErode, config="-c tessedit_char_blacklist=,.!@#$%^&*()_+-=—")
    print(text.strip())
    cv2.imshow("Edge Find", CIErode)
    cv2.waitKey()
    return text.strip()

def ValidateImage(filePath, thresholdLow, thresholdHigh, thresholdLowInc, thresholdHighInc, thickness, target):
    img = cv2.imread(filePath)
    validDate = False
    attempts = 0
    clList = []
    targetDate = []

    for i in target:
        targetDate.append(i)

    # Loop changing the threshold values until a good image is found or 10 attempts have been made.
    while not validDate and attempts < 15:

        thresholdLow = thresholdLow + thresholdLowInc
        thresholdHigh = thresholdHigh + thresholdHighInc
        validDate0 = False
        validDate1 = False
        validDate2 = False
        validDate3 = False
        validDate4 = False
        validDate5 = False
        validDate6 = False
        validCarton = True
        validCartonNo = True
        testText = ""

        # First try with FindRectangle Method
        X, Y, W, H = FindRectangle(img)
        if X == 0 and Y == 0 and W == 0 and H == 0:
            validCarton = False
        else:
            croppedImage = img[Y:Y + H, X:X + W]
            # Increment the threshold by 5 each time
            testText = FindText(croppedImage, thresholdLow, thresholdHigh, thickness)

        # Try with noRect method
        if len(testText) < 10:
            X, Y, W, H = NoRectangleFound(img)
            if X == 0 and Y == 0 and W == 0 and H == 0:
                validCartonNo = False
            else:
                croppedImage = img[Y:Y + H, X:X + W]
                testText = FindText(croppedImage, thresholdLow, thresholdHigh, thickness)


        if not validCarton and not validCartonNo:
            return "Invalid Carton"


        # Use regex to ensure string is good.
        # First create char array
        # Note that Best Before Dates are in the following format
        # DDMMMYY eg - 06DEC21
        # focus on finding correct date rather than the entire string.

        regList = []
        for i in testText:
            regList.append(i)
        closenessScore = 0
        if len(regList) > 6:
            # Pos 0 should be a digit
            if re.match(targetDate[0], regList[0]):
                validDate0 = True
                closenessScore = closenessScore + 1
            # Pos 1 should be a digit
            if re.match(targetDate[1], regList[1]):
                validDate1 = True
                closenessScore = closenessScore + 1
            # Pos 2 should J,F,M,A,M,J,J,A,S,O,N,D
            if re.search(targetDate[2], regList[2]):
                validDate2 = True
                closenessScore = closenessScore + 1
            # Pos 3 should A,E,P,U,C,O
            if re.search(targetDate[3], regList[3]):
                validDate3 = True
                closenessScore = closenessScore + 1
            # Pos 4 should N,B,R,Y,L,G,P,T,V,C
            if re.search(targetDate[4], regList[4]):
                validDate4 = True
                closenessScore = closenessScore + 1
            # Pos 5 should be a digit
            if re.match(targetDate[5], regList[5]):
                validDate5 = True
                closenessScore = closenessScore + 1
            # Pos 6 should be a digit
            if re.match(targetDate[6], regList[6]):
                validDate6 = True
                closenessScore = closenessScore + 1

            if validDate0 and validDate1 and validDate2 and validDate3 and validDate4 and validDate5 and validDate6:
                validDate = True
                return testText
                break
            else:
                validDate = False
                clList.append(closenessScore)

        attempts = attempts + 1

    if not validDate:
        # cv2.imshow("pic", croppedImage)
        # cv2.waitKey()
        CalcThresh(clList, filePath, target)

def CalcThresh(clList, fileName, target):
    thickness = 3
    if len(clList) == 0:
        threshLow = 20
        threshHigh = 60
        threshLowInc = 5
        threshHighInc = 15
    else:
        a = max(clList)
        for i in clList:
            if i == a and a > 4:
                threshLow = i * 5
                threshHigh = i * 15
                threshLowInc = 0
                threshHighInc = 1
            else:
                return "Calc Tresh Fail to find suitable image."
    text = ValidateImage(fileName, threshLow, threshHigh, threshLowInc, threshHighInc, thickness, target)
    return text

text = CalcThresh([], "1.png", "13FEB22")
print("Success the correct text was found in img: {i}".format(i=text))

text = CalcThresh([], "2.png", "27FEB22")
print("Success the correct text was found in img: {i}".format(i=text))

text = CalcThresh([], "3.png", "05JUN22")
print("Success the correct text was found in img: {i}".format(i=text))

text = CalcThresh([], "4.png", "05JUN22")
print("Success the correct text was found in img: {i}".format(i=text))

text = CalcThresh([], "5.png", "05JUN22")
print("Success the correct text was found in img: {i}".format(i=text))

text = CalcThresh([], "6.png", "19JUN22")
print("Success the correct text was found in img: {i}".format(i=text))

text = CalcThresh([], "7.png", "19JUN22")
print("Success the correct text was found in img: {i}".format(i=text))

# Have an image just containing the RoI now :)
# cv2.imshow("Edge Find", img)
# cv2.waitKey()



