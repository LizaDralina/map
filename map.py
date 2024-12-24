
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

image = cv.imread('map5.jpg')
image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

new_image = cv.imread('world.jpg')
new_image_rgb = cv.cvtColor(new_image, cv.COLOR_BGR2RGB)

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
_, thresh = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
closed = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)

contours, _ = cv.findContours(closed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

filtered_contours = [cnt for cnt in contours if cv.contourArea(cnt) > 500]

mask = np.zeros_like(gray)
cv.drawContours(mask, filtered_contours, -1, (255), thickness=cv.FILLED)

new_texture = np.zeros_like(image_rgb)
new_texture[:] = [255, 0, 0]

result = image_rgb.copy()

new_image_resized = cv.resize(new_image_rgb, (image.shape[1], image.shape[0]))

result[mask == 255] = new_image_resized[mask == 255]

plt.figure(figsize=(18, 10))

plt.subplot(2, 3, 1)
plt.title('Оригинал')
plt.imshow(image_rgb)

plt.subplot(2, 3, 2)
plt.title('Бинаризация')
plt.imshow(thresh, cmap='gray')

plt.subplot(2, 3, 3)
plt.title('Закрытие (Closing)')
plt.imshow(closed, cmap='gray')

plt.subplot(2, 3, 4)
plt.title('Контуры')
contour_preview = image_rgb.copy()
cv.drawContours(contour_preview, filtered_contours, -1, (0, 255, 0), thickness=2)
plt.imshow(contour_preview)

plt.subplot(2, 3, 5)
plt.title('Маска карты')
plt.imshow(mask, cmap='gray')

plt.subplot(2, 3, 6)
plt.title('Результат замены')
plt.imshow(result)

plt.tight_layout()
plt.show()
