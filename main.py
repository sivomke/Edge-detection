from PIL import Image
import numpy as np
import random


def rgb_to_greyscale(image: np.ndarray):
    """
    converts RGB image to greyscale
    :param image: RGB image as np.array
    :return: image with all 3 channels set to greyscale value
    """
    height, width = image.shape[0], image.shape[1]
    grey = np.zeros(shape=[height, width, 3], dtype = np.uint8)
    grey[:, :, 0] = 0.2989*image[:, :, 0] + 0.5870*image[:, :, 1] + 0.1140 * image[:, :, 2]
    grey[:, :, 1] = grey[:, :, 0]
    grey[:, :, 2] = grey[:, :, 0]
    return grey


# edge detection
def sobel(image: np.ndarray) -> np.ndarray:
    """
    applies sobel operator to color intensities
    coords: x: rows, y: cols
    :param image:
    :return: result as array
    """
    width = image.shape[1]
    height = image.shape[0]
    result = np.zeros(shape=(height, width), dtype=np.uint8)
    horizontal_mask = np.array([[-1, -2, -1],
                        [0,  0,  0],
                        [1,  2,  1]])
    vertical_mask = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    for x in range(1, height-3):
        for y in range(1, width-3):
            # print("x: {}, y: {}".format(x, y))
            g_x = np.sum(horizontal_mask*image[x-1,1:x+2, y-1:y+2])
            g_y = np.sum(vertical_mask*image[x-1:x+2, y-1:y+2])
            grad = np.sqrt(g_x**2+g_y**2)
            result[x, y] = grad
    return result


# adds salt and pepper noise to greyscaled image

def salt_and_pepper(image: np.ndarray, prob: float) -> np.ndarray:
    width = image.shape[1]
    height = image.shape[0]
    threshold = 1 - prob
    # determine which pixels will be changed
    random_matrix = np.random.random_sample((height, width))
    image[random_matrix < prob] = 0  # change corresponding pixels to black
    image[random_matrix > threshold] = 255  # change corresponding pixels to white
    return image


def median_filter(image: np.ndarray) -> np.ndarray:
    width = image.shape[1]
    height = image.shape[0]
    result = np.zeros(shape=(height, width), dtype=np.uint8)
    for x in range(1, height - 3):
        for y in range(1, width - 3):
            result[x, y] = np.median(image[x - 1:x + 2, y - 1:y + 2])
    return result


def first_order_approx(image: np.ndarray) -> np.ndarray:
    """
    first order intensity approximation  f(i, j)= a*i + b*j + c
    :param image: color intensity
    :return: gradient of first order approximation
    """
    height = image.shape[0]
    width = image.shape[1]
    result = np.zeros(shape=(height, width), dtype=np.uint8)
    a_mask = np.array([[-1, -1], [1, 1]])
    b_mask = np.array([[-1, 1], [-1, 1]])
    for i in range(1, height):
        for j in range(1, width):
            a = .5*np.sum(a_mask*image[i-1:i+1, j-1:j+1])
            b = .5*np.sum(b_mask * image[i - 1:i + 1, j - 1:j + 1])
            result[i,j] = np.sqrt(a**2+b**2)
    return result





def main():
    image = Image.open('puppy.jpg')
    image.show()
    image_arr = np.array(image)
    # red, green, blue channels
    r, g, b = image_arr[:, :, 0], image_arr[:, :, 1], image_arr[:, :, 2]
    greysc = rgb_to_greyscale(image_arr)



    """
    sobel_result = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    sobel_result[:, :, 0] = sobel(image_arr[:, :, 0])
    sobel_result[:, :, 1] = sobel_result[:, :, 0]
    sobel_result[:, :, 2] = sobel_result[:, :, 0]
    result = Image.fromarray(sobel_result, 'RGB')
    result.save('sobel.png')
    result.show()
    """

    """
    first_order = greysc
    first_order[:, :, 0] = first_order_approx(first_order[:, :, 0])
    first_order[:, :, 1] = first_order[:, :, 0]
    first_order[:, :, 2] = first_order[:, :, 0]
    result = Image.fromarray(first_order, 'RGB')
    result.save('first_order.png')
    result.show()
    """


    """
    salt_pepper = greysc
    salt_pepper[:, :, 0] = salt_and_pepper(salt_pepper[:, :, 0], 0.05)
    salt_pepper[:, :, 1] = salt_pepper[:, :, 0]
    salt_pepper[:, :, 2] = salt_pepper[:, :, 0]
    result = Image.fromarray(salt_pepper, 'RGB')
    result.save('salt_and_pepper.png')
    result.show()
    """

    """
    median = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    median[:, :, 0] = median_filter(salt_pepper[:, :, 0])
    median[:, :, 1] = median[:, :, 0]
    median[:, :, 2] = median[:, :, 0]
    result = Image.fromarray(median, 'RGB')
    result.save('median_filter.png')
    result.show()
    """

main()


