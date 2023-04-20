import json

import numpy as np
from skimage.exposure import histogram
from matplotlib.pyplot import title, tight_layout
from skimage.io import imread, imsave, imshow, show
from matplotlib import pyplot as plt
from skimage import data, exposure  # , img_as_float

path = 'images\\test.jpg'
image = imread(path)
g_min = 0
g_max = 255


def add_plot(fig, rows, columns, cell, x, y, _title):
    plt.rcParams['font.size'] = '7'
    fig.add_subplot(rows, columns, cell)
    title(_title)
    plt.plot(x, y, color='teal', linewidth=2)


def add_picture(fig, rows, columns, cell, image, _title):
    plt.rcParams['font.size'] = '7'
    fig.add_subplot(rows, columns, cell)
    title(_title)
    imshow(image)


def threshold_processing():
    fig = plt.figure(figsize=(12, 5))
    add_picture(fig, 2, 3, 1, image, 'The original image')
    hist, bins = histogram(image)
    add_plot(fig, 2, 3, 2, bins, hist, 'The original histogram')

    # threshold(порог) for image
    threshold = 125
    threshold_processing_image = image > threshold
    threshold_processing_image = (threshold_processing_image * 255).astype(np.uint8)

    add_picture(fig, 2, 3, 4, threshold_processing_image, 'The image after threshold processing')
    hist, bins = histogram(threshold_processing_image)
    add_plot(fig, 2, 3, 5, bins, hist, 'The histogram after threshold processing')

    x = np.sort(image.ravel())
    y = np.sort(threshold_processing_image.ravel())
    add_plot(fig, 2, 3, 6, x, y, 'The function element - by - element conversion')

    show()


def contrasting():
    fig = plt.figure(figsize=(12, 5))
    add_picture(fig, 2, 3, 1, image, 'The original image')

    hist, bins = histogram(image)
    add_plot(fig, 2, 3, 2, bins, hist, 'The original histogram')

    # [fmin,fmax] - real range
    f_min = image.min()
    f_max = image.max()

    a = (g_max - g_min) / (f_max - f_min)
    b = (g_min * f_max - g_max * f_min) / (f_max - f_min)

    contrasted_image = (a * image + b).astype(np.uint8)
    add_picture(fig, 2, 3, 4, contrasted_image, 'The contrasted image')

    hist, bins = histogram(contrasted_image)
    add_plot(fig, 2, 3, 5, bins, hist, 'The histogram after contrasting processing')

    x = np.sort(image.ravel())
    y = np.sort(contrasted_image.ravel())
    title('The function element - by - element conversion')
    add_plot(fig, 2, 3, 6, x, y, 'The function element - by - element conversion')

    show()


# 3) Equalization
def equalization():
    fig = plt.figure(figsize=(17, 7))
    add_picture(fig, 3, 4, 1, image, 'The original image')

    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cumsum = hist.cumsum()
    F = (cumsum - cumsum.min()) / (cumsum.max() - cumsum.min())
    g = (g_max - g_min) * F + g_min
    equalization_image = g[image]
    equalization_image = equalization_image.astype(np.uint8)
    # self-written = самописной
    add_picture(fig, 3, 4, 2, equalization_image, 'The self-written equalization image')

    standart_equalization_image = exposure.equalize_hist(image)
    add_picture(fig, 3, 4, 3, standart_equalization_image, 'Image after standart equalization')

    hist, bins = histogram(image)
    add_plot(fig, 3, 4, 5, bins, hist, 'The histogram of original image')

    x = np.arange(0, 256, 1)
    add_plot(fig, 3, 4, 6, x, F, 'Graph of the cumulative distribution function of brightness before')

    hist, bins = np.histogram(equalization_image.flatten(), 256, [0, 256])
    cumsum1 = hist.cumsum()
    F_after_equalization = (cumsum1 - cumsum1.min()) / (cumsum1.max() - cumsum1.min())
    g_after_equalization = (g_max - g_min) * F + g_min
    add_plot(fig, 3, 4, 7, x, F_after_equalization, 'Graph of the cumulative distribution function of brightness after')

    add_plot(fig, 3, 4, 8, x, g[x], 'The function element - by - element conversion')

    hist, bins = histogram(equalization_image)
    add_plot(fig, 3, 4, 9, bins, hist, 'The histogram of self-written equalization image')

    hist, bins = histogram(standart_equalization_image)
    add_plot(fig, 3, 4, 10, bins, hist, 'The histogram of standart equalization image')

    plt.tight_layout()
    show()


if __name__ == '__main__':
    threshold_processing()
    # contrasting()
    # equalization()
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    print(bins)
    print('\n')
    print(hist)
