import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

edited = cv.imread('edited.jpg')
rio = cv.imread('rio.jpg')
bodo = cv.imread('bodo.jpg')
budapest = cv.imread('budapest.jpg')
ny = cv.imread('ny.jpg')
prague = cv.imread('prague.jpg')
differences_list = []
images = [
    [rio, 700, 1400],
    [bodo, 0, 1100],
    [prague, 250, 950],
    [budapest, 520, 1280],
    [ny, 250, 950]  # Total disaster
]

dif = cv.absdiff(cv.imread('org.jpg'), edited)
dif = cv.cvtColor(dif, cv.COLOR_BGR2GRAY)
ret, dif = cv.threshold(dif, 36, 255, cv.THRESH_BINARY)
#dif = cv.Canny(dif, 50, 100)
dilate = cv.dilate(dif, np.ones((3, 3), np.uint8), iterations=1)
result = cv.bitwise_and(edited, edited, mask=dilate)
edited = cv.cvtColor(edited, cv.COLOR_BGR2RGB)
result = cv.cvtColor(result, cv.COLOR_BGR2RGB)
contours, hierarchy = cv.findContours(dilate, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for i in range(len(contours)):
    x, y, dx, dy = cv.boundingRect(contours[i])
    differences_list.append(result[y:y + dy, x:x + dx])
    cv.rectangle(edited, (x, y), (x + dx, y + dy), (0, 255, 0), 5)
    plt.subplot(2, 4, i + 1)
    plt.imshow(differences_list[i])
    plt.axis('off')

plt.show()
plt.imshow(edited)
plt.axis('off')
plt.show()

#########################
### Zadanie dodatkowe ###
#########################

template = differences_list[7]
template_blur = cv.GaussianBlur(template, (3, 3), 0)
template_edges = cv.Canny(template, 50, 100)
template_contours, _ = cv.findContours(template_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for image in images:
    image_blur = cv.GaussianBlur(image[0], (3, 3), 0)
    image_edges = cv.Canny(image[0], image[1], image[2])
    image_contours, _ = cv.findContours(image_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for image_contour in image_contours:
        for template_contour in template_contours:
            if abs(cv.contourArea(image_contour) - cv.contourArea(template_contour)) < 50 and cv.contourArea(template_contour) > 50:
                epsilon = 0.2 * cv.arcLength(image_contour, True)
                approx = cv.approxPolyDP(image_contour, epsilon, True)
                template_epsilon = 0.2 * cv.arcLength(template_contour, True)
                template_approx = cv.approxPolyDP(template_contour, template_epsilon, True)
                if len(approx) == len(template_approx):
                    x, y, dx, dy = cv.boundingRect(image_contour)
                    cv.rectangle(image[0], (x, y), (x + dx, y + dy), (0, 255, 0), 5)

    image[0] = cv.cvtColor(image[0], cv.COLOR_BGR2RGB)
    plt.imshow(image[0])
    plt.axis('off')
    plt.show()
