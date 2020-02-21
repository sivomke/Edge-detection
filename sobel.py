from PIL import Image
import numpy as np


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
            g_x = np.sum(horizontal_mask*image[x-1:x+2, y-1:y+2])
            g_y = np.sum(vertical_mask*image[x-1:x+2, y-1:y+2])
            grad = np.sqrt(g_x**2+g_y**2)
            result[x, y] = grad
    return result


def main():
    image = Image.open('puppy.jpg')
    # image.show()
    image_arr = np.array(image)
    height = image_arr.shape[0]
    width = image_arr.shape[1]
    # red, green, blue channels
    r, g, b = image_arr[:, :, 0], image_arr[:, :, 1], image_arr[:, :, 2]
    greysc = rgb_to_greyscale(image_arr)

    sobel_result = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    sobel_result[:, :, 0] = sobel(image_arr[:, :, 0])
    sobel_result[:, :, 1] = sobel_result[:, :, 0]
    sobel_result[:, :, 2] = sobel_result[:, :, 0]
    result = Image.fromarray(sobel_result, 'RGB')
    result.save('sobel.png')
    result.show()







