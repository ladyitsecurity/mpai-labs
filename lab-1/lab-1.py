import json

import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import title, tight_layout
from skimage.io import imread, imsave, imshow, show
from matplotlib import pyplot as plt
from skimage import data, exposure, img_as_float

settings = {'read_path': 'C:\\Users\\Zver\\PycharmProjects\\mpai_lab1\\images\\test.jpg',
            'save_path': 'new.tif'
            }

# open file  and serialize settings in this file
with open('settings.json', 'w') as fp:
    json.dump(settings, fp)

with open('settings.json') as info_data:
    json_data = json.load(info_data)

path = json_data['read_path']
image = imread(path)

# 1) Threshold processing

fig = plt.figure(figsize=(16, 10))
fig.add_subplot(2, 3, 1)
title('The original image')
imshow(image)

fig.add_subplot(2, 3, 2)
title('The original histogram')
hist, bins = histogram(image)
plt.plot(bins, hist)

# threshold(порог) for image
threshold = 125
threshold_processing_image = image > threshold
threshold_processing_image = (threshold_processing_image * 255).astype(np.uint8)

fig.add_subplot(2, 3, 4)
title('The image after threshold processing')
imshow(threshold_processing_image)

fig.add_subplot(2, 3, 5)
title('The histogram after threshold processing')
hist, bins = histogram(threshold_processing_image)
plt.plot(bins, hist)

fig.add_subplot(2, 3, 6)
# function ravel https://pyprog.pro/array_manipulation/ravel.html
# function ravel return a contiguous flattened array(сжатый до одной оси массив).
x = np.sort(image.ravel())
y = np.sort(threshold_processing_image.ravel())
title('The function element - by - element conversion')
plt.plot(x, y)

show()

# 2) Contrasting
fig = plt.figure(figsize=(16, 10))
fig.add_subplot(2, 3, 1)
title('The original image')
imshow(image)

fig.add_subplot(2, 3, 2)
title('The original histogram')
hist, bins = histogram(image)
plt.plot(bins, hist)

# [gmin, gmax] - required(acceptable) range
g_min = 0
g_max = 255
# [fmin,fmax] - real range
f_min = image.min()
f_max = image.max()

a = (g_max - g_min) / (f_max - f_min)
b = (g_min * f_max - g_max * f_min) / (f_max - f_min)

contrasted_image = (a * image + b).astype(np.uint8)
fig.add_subplot(2, 3, 4)
title('The contrasted image')
imshow(contrasted_image)

fig.add_subplot(2, 3, 5)
title('The histogram after contrasting processing')
hist, bins = histogram(contrasted_image)
plt.plot(bins, hist)

fig.add_subplot(2, 3, 6)
x = np.sort(image.ravel())
y = np.sort(contrasted_image.ravel())
title('The function element - by - element conversion')
plt.plot(x, y)

show()

# 3) Equalization

fig = plt.figure(figsize=(25, 10))
fig.add_subplot(3, 4, 1)
title('The original image')
imshow(image)

hist, bins = np.histogram(image.flatten(), 256, [0, 256])
cumsum = hist.cumsum()
F = (cumsum - cumsum.min()) / (cumsum.max() - cumsum.min())
g = (g_max - g_min) * F + g_min
equalization_image = g[image]
equalization_image = equalization_image.astype(np.uint8)
fig.add_subplot(3, 4, 2)
# self-written = самописной
title('The self-written equalization image')
imshow(equalization_image)

standart_equalization_image = exposure.equalize_hist(image)
fig.add_subplot(3, 4, 3)
title('Image after standart equalization')
imshow(standart_equalization_image)

x = np.arange(0, 256, 1)

fig.add_subplot(3, 4, 5)
title('Graph of the cumulative distribution function of brightness before')
plt.plot(x, F)

hist, bins = np.histogram(equalization_image.flatten(), 256, [0, 256])
cumsum1 = hist.cumsum()
F_after_equalization = (cumsum1 - cumsum1.min()) / (cumsum1.max() - cumsum1.min())
g_after_equalization = (g_max - g_min) * F + g_min
fig.add_subplot(3, 4, 6)
title('Graph of the cumulative distribution function of brightness after')
plt.plot(x, F_after_equalization)

fig.add_subplot(3, 4, 7)
title('The histogram of original image')
hist, bins = histogram(image)
plt.plot(bins, hist)

fig.add_subplot(3, 4, 8)
title('The histogram of self-written equalization image')
hist, bins = histogram(equalization_image)
plt.plot(bins, hist)

fig.add_subplot(3, 4, 9)
title('The histogram of standart equalization image')
hist, bins = histogram(standart_equalization_image)
plt.plot(bins, hist)

fig.add_subplot(3, 4, 10)
title('The function element - by - element conversion')
plt.plot(x, g[x])

plt.tight_layout()
show()

hist, bins = np.histogram(image.flatten(), 256, [0, 256])
print(bins)
print('\n')
print(hist)