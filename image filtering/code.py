import numpy as np
import math
from numpy.fft import fft2, ifft2, fftshift, ifftshift


def getGaussianKernel(rows, cols, sigma):
    centerX = int(rows/2) + 1 if rows % 2 == 1 else int(rows/2)
    centerY = int(cols/2) + 1 if cols % 2 == 1 else int(cols/2)
    gaussian = lambda i,j: math.exp(-1.0 * ((i - centerX)**2 + (j - centerY)**2) / (2 * sigma**2))

    return np.array([[gaussian(i,j) for j in range(cols)] for i in range(rows)])


def myfft_imfilter(image, filter):
    def imfilterDFT(image, filter):
        shiftedDFT = fftshift(fft2(image))
        filteredDFT = shiftedDFT * filter
        return ifft2(ifftshift(filteredDFT))
    
    if image.shape[0] != filter.shape[0] and image.shape[1] != filter.shape[1]:
        raise AttributeError('The shape of filter should be the same as image.')

    channels = []
    for k in range(image.shape[2]):
        channels.append(imfilterDFT(image, filter))
    
    return np.dstack(channels)


def my_imfilter(image, filter):
    """
    Apply a filter to an image. Return the filtered image.

    Args
    - image: numpy nd-array of dim (m, n, c)
    - filter: numpy nd-array of dim (k, k)
    Returns
    - filtered_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You may not use any libraries that do the work for you. Using numpy to work
    with matrices is fine and encouraged. Using opencv or similar to do the
    filtering for you is not allowed.
    - I encourage you to try implementing this naively first, just be aware that
    it may take an absurdly long time to run. You will need to get a function
    that takes a reasonable amount of time to run so that the TAs can verify
    your code works.
    - Remember these are RGB images, accounting for the final image dimension.
    """

    assert filter.shape[0] % 2 == 1
    assert filter.shape[1] % 2 == 1

    filterH, filterW = filter.shape[0], filter.shape[1]
    padded_image = np.pad(image, ((filterH//2, filterH//2), 
                         (filterW//2, filterW//2), (0, 0)), "symmetric")
    filtered_image = np.zeros((padded_image.shape[0],
                               padded_image.shape[1],
                               padded_image.shape[2]))

    for k in range(image.shape[2]):
        for H in range(filterH//2, padded_image.shape[0] - filterH//2):
            for W in range(filterW//2, padded_image.shape[1] - filterW//2):
                padded_zone = padded_image[H - filterH//2: H + filterH//2 + 1,
                                           W - filterW//2: W + filterW//2 + 1,
                                           k]

                filtered_image[H, W, k] = np.sum(np.multiply(filter, padded_zone))

    filtered_image = filtered_image[filterH//2: filtered_image.shape[0] - filterH//2,
                           filterW//2: filtered_image.shape[1] - filterW//2]
    return np.clip(filtered_image, 0, 1, out = filtered_image)


def create_hybrid_image(image1, image2, filter):
    """
    Takes two images and creates a hybrid image. Returns the low
    frequency content of image1, the high frequency content of
    image 2, and the hybrid image.

    Args
    - image1: numpy nd-array of dim (m, n, c)
    - image2: numpy nd-array of dim (m, n, c)
    Returns
    - low_frequencies: numpy nd-array of dim (m, n, c)
    - high_frequencies: numpy nd-array of dim (m, n, c)
    - hybrid_image: numpy nd-array of dim (m, n, c)

    HINTS:
    - You will use your my_imfilter function in this function.
    - You can get just the high frequency content of an image by removing its low
    frequency content. Think about how to do this in mathematical terms.
    - Don't forget to make su
    - If you want to use images with different dimensions, you should resize them
    in the notebook code.re the pixel values are >= 0 and <= 1. This is known
    as 'clipping'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    blurry_image1 = my_imfilter(image1, filter)
    blurry_image2 = my_imfilter(image2, filter)

    low_frequencies = blurry_image1
    high_frequencies = image2 - blurry_image2

    # define the hybrid coefficient
    p = 0.5
    hybrid_image = p * low_frequencies + (1 - p) * high_frequencies
    hybrid_image = np.clip(hybrid_image, 0, 1, hybrid_image)
    return low_frequencies, high_frequencies, hybrid_image