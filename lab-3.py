import scipy.ndimage
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.pyplot import show
from skimage.util import random_noise
from scipy.signal import convolve2d
from scipy.ndimage import median_filter
from skimage.io import imshow

# useful link
# https://scikit-image.org/docs/dev/api/skimage.util.html#skimage.util.random_noise

def chess_board(w, h, cell, min, max):
    """
    :param  w       weight of chess board
    :param  h       длина of chess board
    :param  cell    size of sell
    :param  min     minimum brightness value
    :param  max     maximum brightness value
    """
    board = [[max, min] * int(w / cell / 2), [min, max] * int(w / cell / 2)] * int(h / cell / 2)
    board = (np.kron(board, np.ones((cell, cell))))
    return board


def white_noise(d, image, mask, footprint, w, h):
    """
    the function demonstrates the restoration of a noisy white image using a linear and median filter
    :param d:           signal-to-noise ratio
    :param image:       input image
    :param mask:        mask for linear filter
    :param footprint:   mask for median filter
    :param w:           weight of input image
    :param h:           height of input image
    """
    dispersion_of_image = np.var(image)

    # var - Variance of random distribution. Used in ‘gaussian’ and ‘speckle’.
    # Note: variance = (standard deviation) ** 2. Default : 0.01
    noise_image = random_noise(image, var=dispersion_of_image / d)
    noise = noise_image - image
    dispersion_of_noise = np.var(noise)
    print('Дисперсия аддитивного белого шума = ', dispersion_of_noise)

    median_image = median_filter(footprint, noise_image, image, w, h)

    linear_image = linear_smoothing_filter(mask, noise_image, image, w, h)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"d^2 = {d}")

    fig.add_subplot(2, 3, 1)
    plt.title('Исходное изображение')
    imshow(image, cmap='gray')

    fig.add_subplot(2, 3, 2)
    plt.title('Аддитивный белый шум')
    imshow(noise, cmap='gray')

    fig.add_subplot(2, 3, 3)
    plt.title('Зашумленное изображение')
    imshow(noise_image, cmap='gray')

    fig.add_subplot(2, 3, 4)
    plt.title('Изображение фильтрованное медианным')
    imshow(median_image, cmap='gray')

    fig.add_subplot(2, 3, 5)
    plt.title('Изображение фильтрованное линейным')
    imshow(linear_image, cmap='gray')

    show()


def impulse_noise(image, p, mask, footprint, w, h):
    """
    the function demonstrates the restoration of a impulse noise image using a linear and median filter
    :param image:       input image
    :param p:
    :param mask:        mask for linear filter
    :param footprint:   mask for median filter
    :param w:           weight of input image
    :param h:           height of input image
    :return:
    """

    # amount - Proportion of image pixels to replace with noise on range [0, 1].
    # Used in ‘salt’, ‘pepper’, and ‘salt & pepper’. Default : 0.05
    noise_image = random_noise(image, mode='s&p', amount=p)
    noise = image - noise_image
    dispersion_of_noise = np.var(noise)
    print('Дисперсия импульсного шума = ', dispersion_of_noise)

    median_image = median_filter(footprint, noise_image, image, w, h)

    linear_image = linear_smoothing_filter(mask, noise_image, image, w, h)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"p = {p}")

    fig.add_subplot(2, 3, 1)
    plt.title('Исходное изображение')
    imshow(image, cmap='gray')

    fig.add_subplot(2, 3, 2)
    plt.title('Импульсный шум')
    imshow(noise, cmap='gray')

    fig.add_subplot(2, 3, 3)
    plt.title('Зашумленное изображение')
    imshow(noise_image, cmap='gray')

    fig.add_subplot(2, 3, 4)
    plt.title('Изображение фильтрованное медианным')
    imshow(median_image, cmap='gray')

    fig.add_subplot(2, 3, 5)
    plt.title('Изображение фильтрованное линейным')
    imshow(linear_image, cmap='gray')

    show()


3


def median_filter(footprint, noise_image, image, w, h):
    median_image = scipy.ndimage.median_filter(noise_image, footprint=footprint)
    dispersion_of_error = np.power(np.sum(median_image) - np.sum(image), 2) / (w * h)
    print('Дисперсия ошибки фильтрации медианным = ', dispersion_of_error)
    noise_suppression_ratio = np.mean(np.power(median_image - image, 2)) / np.mean(np.power(noise_image - image, 2))
    print('Коэффициент подавления шума медианным = ', noise_suppression_ratio)
    return median_image


def linear_smoothing_filter(mask, noise_image, image, w, h):
    linear_image = convolve2d(noise_image, mask, boundary='symm', mode='same')
    dispersion_error = np.power(np.sum(linear_image) - np.sum(image), 2) / (w * h)
    print('Дисперсия ошибки фильтрации линейным = ', dispersion_error)
    noise_suppression_ratio = np.mean(np.power(linear_image - image, 2)) / np.mean(np.power(noise_image - image, 2))
    print("Коэффициент подавления шума линейным = ", noise_suppression_ratio)
    print()
    return linear_image


if __name__ == '__main__':
    board_width = 128
    board_height = 128
    cell_size = 16
    min_bright = 96 / 255
    max_bright = 160 / 255

    img = chess_board(board_width, board_height, cell_size, min_bright, max_bright)

    footprint = np.array([[0, 1, 0], [1, 3, 1], [0, 1, 0]])
    # footprint = np.array([[1, 1, 1], [1, 3, 1], [1, 1, 1]])
    # footprint = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    # footprint = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    mask = 1 / 9 * np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
    # mask = 1/10 * np.array([[1, 1, 1], [1, 2, 1], [1, 1, 1]])
    # mask = 1/16 * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    dispersion_image = np.var(img)
    print('Дисперсия исходного изображения = ', dispersion_image)
    print()

    white_noise(1, img, mask, footprint, board_width, board_height)
    white_noise(10, img, mask, footprint, board_width, board_height)

    impulse_noise(img, 0.1, mask, footprint, board_width, board_height)
    impulse_noise(img, 0.3, mask, footprint, board_width, board_height)
