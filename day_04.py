import cv2
import numpy as np
import matplotlib.pyplot as plt


img = cv2.imread(r'c:\Users\HP\Documents\Personal Details\KAMISHKA PHOTO 68 KB.jpg')
print(type(img))
print(img.shape)

cv2.imshow("Photos", img)
cv2.waitKey(0)

img_resize = cv2.resize(img, (400, 500))
cv2.imshow("Photos", img_resize)
cv2.waitKey(0)

img_flip_horizontal = cv2.flip(img_resize, 0)#horizontalflip
cv2.imshow("Photos", img_flip_horizontal)
cv2.waitKey(0)

img_flip_vertical = cv2.flip(img_resize, 1)#verticalflip
cv2.imshow("Photos", img_flip_vertical)
cv2.waitKey(0)

img_crop = img[100:300, 200:500]
cv2.imshow("Photos", img_crop)
cv2.waitKey(0)

cv2.destroyAllWindows()

