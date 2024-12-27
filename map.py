import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


image = cv.imread('map6.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

new_image = cv.imread('world.jpg')
new_image_rgb = cv.cvtColor(new_image, cv.COLOR_BGR2RGB)

original_points = np.array([
    [235, 188],
    [970, 110],
    [940, 670],
    [210, 600]
], dtype="float32")

new_points = np.array([
    [0, 0],
    [new_image.shape[1], 0],
    [new_image.shape[1], new_image.shape[0]],
    [0, new_image.shape[0]]
], dtype="float32")

homography_matrix, _ = cv.findHomography(new_points, original_points)

warped_new_image = cv.warpPerspective(new_image_rgb, homography_matrix, (image.shape[1], image.shape[0]))

mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
cv.fillPoly(mask, [original_points.astype(np.int32)], 255)

result = image_rgb.copy()
result[mask == 255] = warped_new_image[mask == 255]

plt.figure(figsize=(18, 10))

plt.subplot(1, 3, 1)
plt.title('Оригинал')
plt.imshow(image_rgb)

plt.subplot(1, 3, 2)
plt.title('Новая текстура')
plt.imshow(warped_new_image)

plt.subplot(1, 3, 3)
plt.title('Результат замены')
plt.imshow(result)

plt.tight_layout()
plt.show()


