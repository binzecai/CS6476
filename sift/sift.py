import numpy as np
import cv2


def get_features(image, x, y, feature_width, scales=None):
    """
    To start with, you might want to simply use normalized patches as your
    local feature. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) descriptor should have:
    (1) a 4x4 grid of cells, each feature_width/4. It is simply the
        terminology used in the feature literature to describe the spatial
        bins where gradient distributions will be described.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4 x 4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length.

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like feature can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Args:
    -   image: A numpy array of shape (m,n) or (m,n,c). can be grayscale or color, your choice
    -   x: A numpy array of shape (k,), the x-coordinates of interest points
    -   y: A numpy array of shape (k,), the y-coordinates of interest points
    -   feature_width: integer representing the local feature width in pixels.
            You can assume that feature_width will be a multiple of 4 (i.e. every
                cell of your local SIFT-like feature will have an integer width
                and height). This is the initial window size we examine around
                each keypoint.
    -   scales: Python list or tuple if you want to detect and describe features
            at multiple scales

    You may also detect and describe features at particular orientations.

    Returns:
    -   fv: A numpy array of shape (k, feat_dim) representing a feature vector.
            "feat_dim" is the feature_dimensionality (e.g. 128 for standard SIFT).
            These are the computed features.
    """
    assert image.ndim == 2, 'Image must be grayscale'
    #############################################################################
    # TODO: YOUR CODE HERE                                                      #
    # If you choose to implement rotation invariance, enabling it should not    #
    # decrease your matching accuracy.                                          #
    #############################################################################

    assert image.ndim == 2, 'Image must be grayscale'
    gaussian_filter = cv2.getGaussianKernel(ksize=9, sigma=2)
    image = cv2.filter2D(image, -1, gaussian_filter*gaussian_filter.T)

    sobelx = cv2.Sobel(image,cv2.CV_64F,1,0)
    sobely = cv2.Sobel(image,cv2.CV_64F,0,1)

    mag = np.sqrt(sobelx**2.0 + sobely**2.0)
    ori = np.arctan2(sobely, sobelx) *180/np.pi+180

    gaussian_filter = cv2.getGaussianKernel(feature_width,feature_width)

    num_bins = 8
    bin_size = 360/num_bins
    bins_shape = [i*bin_size for i in range(num_bins+1)]

    fv = np.zeros((x.shape[0],128))
    for idx in range(0, len(x)):
        curr_x = y[idx]
        curr_y = x[idx]

        radii = feature_width//2
        mag_cell = mag[int(curr_x - radii) : int(curr_x+radii), int(curr_y - radii) : int(curr_y+radii)]
        ori_cell = ori[int(curr_x - radii) : int(curr_x+radii), int(curr_y - radii) : int(curr_y+radii)]
        mag_cell = cv2.filter2D(mag_cell,-1,gaussian_filter)

        curr_fv = []
        # print (mag_cell.shape)
        if (np.size(mag_cell) == 256):
            for i in range(0,4):
                for j in range(0,4):
                    #gradient magnitude and directions in cell
                    grad_mag_cell = mag_cell[i*4: i*4+4, j*4:j*4+4].ravel()
                    grad_dir_cell = ori_cell[i*4: i*4+4, j*4:j*4+4].ravel()
                    cell_ori = np.histogram(grad_dir_cell, bins=bins_shape, weights=grad_mag_cell)
                    curr_fv.extend(cell_ori[0].tolist())
            fv[idx][:] = curr_fv
    fv = fv/np.linalg.norm(fv)
    fv = np.clip(fv, 0, 0.2)

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return fv
