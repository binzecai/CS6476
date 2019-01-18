import numpy as np
import cyvlfeat as vlfeat
from utils import *
import os.path as osp
from glob import glob
from random import shuffle
from IPython.core.debugger import set_trace
from sklearn.svm import LinearSVC


def get_positive_features(train_path_pos, feature_params):
    """
    This function should return all positive training examples (faces) from
    36x36 images in 'train_path_pos'. Each face should be converted into a
    HoG template according to 'feature_params'.

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features

    Args:
    -   train_path_pos: (string) This directory contains 36x36 face images
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slower because the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time
            (although you don't have to make the detector step size equal a
            single HoG cell).

    Returns:
    -   feats: N x D matrix where N is the number of faces and D is the template
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    positive_files = glob(osp.join(train_path_pos, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################
    
    n_cell = np.ceil(win_size/cell_size).astype('int')
    feats = np.random.rand(len(positive_files), n_cell*n_cell*31)
    for i in range(len(positive_files)):
        gray_img = load_image_gray(positive_files[i])
        hog = vlfeat.hog.hog(gray_img, cell_size)
        feats[i, :] = hog.reshape(1, n_cell*n_cell*31)
    ###########################################################################
    #                             END OF YOUR CODE                            #    
    ###########################################################################

    return feats

def get_random_negative_features(non_face_scn_path, feature_params, num_samples):
    """
    This function should return negative training examples (non-faces) from any
    images in 'non_face_scn_path'. Images should be loaded in grayscale because
    the positive training data is only available in grayscale (use
    load_image_gray()).

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   num_samples: number of negatives to be mined. It is not important for
            the function to find exactly 'num_samples' non-face features. For
            example, you might try to sample some number from each image, but
            some images might be too small to find enough.

    Returns:
    -   N x D matrix where N is the number of non-faces and D is the feature
            dimensionality, which would be (feature_params['template_size'] /
            feature_params['hog_cell_size'])^2 * 31 if you're using the default
            hog parameters.
    """
    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    n_cell = np.ceil(win_size/cell_size).astype('int')
#   feats = np.random.rand(len(negative_files), n_cell*n_cell*31)
    feats = []
    samples_per_image = num_samples / len(negative_files)
    np.random.seed(10)
    if num_samples > 5000:
            samples_per_image += 10

    for i in range(len(negative_files)):
        gray_img = load_image_gray(negative_files[i])
        hogs = vlfeat.hog.hog(gray_img, cell_size)
        dim1, dim2, dim3 = hogs.shape
        for j in range(int(samples_per_image)):
                idx = np.random.randint(dim1)
                idy = np.random.randint(dim2)
                if (idx + n_cell < dim1) and (idy + n_cell < dim2): # dimension overflow
                        hog = hogs[idx:idx+n_cell, idy:idy+n_cell, :dim3] 
                        feats.append(hog.ravel())
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    feats = np.array(feats, dtype=np.float64)
    return feats

def train_classifier(features_pos, features_neg, C):
    """
    This function trains a linear SVM classifier on the positive and negative
    features obtained from the previous steps. We fit a model to the features
    and return the svm object.

    Args:
    -   features_pos: N X D array. This contains an array of positive features
            extracted from get_positive_feats().
    -   features_neg: M X D array. This contains an array of negative features
            extracted from get_negative_feats().

    Returns:
    -   svm: LinearSVC object. This returns a SVM classifier object trained
            on the positive and negative features.
    """
    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    svm = LinearSVC(random_state=0, tol=1e-3, loss='hinge', C=C)
    feats = np.vstack((features_pos, features_neg))
    labels = np.hstack((np.ones(len(features_pos)), -np.ones(len(features_neg))))
    svm.fit(feats, labels)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return svm

def mine_hard_negs(non_face_scn_path, svm, feature_params):
    """
    This function is pretty similar to get_random_negative_features(). The only
    difference is that instead of returning all the extracted features, you only
    return the features with false-positive prediction.

    Useful functions:
    -   vlfeat.hog.hog(im, cell_size): computes HoG features
    -   svm.predict(feat): predict features

    Args:
    -   non_face_scn_path: string. This directory contains many images which
            have no faces in them.
    -   feature_params: dictionary of HoG feature computation parameters. See
            the documentation for get_positive_features() for more information.
    -   svm: LinearSVC object

    Returns:
    -   N x D matrix where N is the number of non-faces which are
            false-positive and D is the feature dimensionality.
    """

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)

    negative_files = glob(osp.join(non_face_scn_path, '*.jpg'))

    ###########################################################################
    #                           TODO: YOUR CODE HERE                          #
    ###########################################################################

    n_cell = np.ceil(win_size/cell_size).astype('int')
    feats = []
    num_samples = 5000
    samples_per_image = num_samples / len(negative_files)
    np.random.seed(10)

    for i in range(len(negative_files)):
        gray_img = load_image_gray(negative_files[i])
        hogs = vlfeat.hog.hog(gray_img, cell_size)
        dim1, dim2, dim3 = hogs.shape
        for j in range(int(samples_per_image)):
                idx = np.random.randint(dim1)
                idy = np.random.randint(dim2)
                if (idx + n_cell < dim1) and (idy + n_cell < dim2): # dimension overflow
                        hog = hogs[idx:idx+n_cell, idy:idy+n_cell, :dim3]
                        if svm.predict(np.array([hog.ravel()])) > 0:
                                feats.append(hog.ravel())

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    feats = np.array(feats, dtype=np.float64)
    return feats

def run_detector(test_scn_path, svm, feature_params, verbose=False):
    """
    This function returns detections on all of the images in a given path. You
    will want to use non-maximum suppression on your detections or your
    performance will be poor (the evaluation counts a duplicate detection as
    wrong). The non-maximum suppression is done on a per-image basis. The
    starter code includes a call to a provided non-max suppression function.

    The placeholder version of this code will return random bounding boxes in
    each test image. It will even do non-maximum suppression on the random
    bounding boxes to give you an example of how to call the function.

    Your actual code should convert each test image to HoG feature space with
    a _single_ call to vlfeat.hog.hog() for each scale. Then step over the HoG
    cells, taking groups of cells that are the same size as your learned
    template, and classifying them. If the classification is above some
    confidence, keep the detection and then pass all the detections for an
    image to non-maximum suppression. For your initial debugging, you can
    operate only at a single scale and you can skip calling non-maximum
    suppression. Err on the side of having a low confidence threshold (even
    less than zero) to achieve high enough recall.

    Args:
    -   test_scn_path: (string) This directory contains images which may or
            may not have faces in them. This function should work for the
            MIT+CMU test set but also for any other images (e.g. class photos).
    -   svm: A trained sklearn.svm.LinearSVC object
    -   feature_params: dictionary of HoG feature computation parameters.
        You can include various parameters in it. Two defaults are:
            -   template_size: (default 36) The number of pixels spanned by
            each train/test template.
            -   hog_cell_size: (default 6) The number of pixels in each HoG
            cell. template size should be evenly divisible by hog_cell_size.
            Smaller HoG cell sizes tend to work better, but they make things
            slowerbecause the feature dimensionality increases and more
            importantly the step size of the classifier decreases at test time.
    -   verbose: prints out debug information if True

    Returns:
    -   bboxes: N x 4 numpy array. N is the number of detections.
            bboxes(i,:) is [x_min, y_min, x_max, y_max] for detection i.
    -   confidences: (N, ) size numpy array. confidences(i) is the real-valued
            confidence of detection i.
    -   image_ids: List with N elements. image_ids[i] is the image file name
            for detection i. (not the full path, just 'albert.jpg')
    """
    im_filenames = sorted(glob(osp.join(test_scn_path, '*.jpg')))
    bboxes = np.empty((0, 4))
    confidences = np.empty(0)
    image_ids = []

    # number of top detections to feed to NMS
    topk = 200

    # params for HOG computation
    win_size = feature_params.get('template_size', 36)
    cell_size = feature_params.get('hog_cell_size', 6)
#     scale_factor = feature_params.get('scale_factor', 0.65)
    template_size = int(win_size / cell_size)

    for idx, im_filename in enumerate(im_filenames):
        print('Detecting faces in {:s}'.format(im_filename))
        im = load_image_gray(im_filename)
        im_id = osp.split(im_filename)[-1]
        im_shape = im.shape
        # create scale space HOG pyramid and return scores for prediction

        #######################################################################
        #                        TODO: YOUR CODE HERE                         #
        #######################################################################

        # cur_x_min = (np.random.rand(15,1) * im_shape[1]).astype('int')
        # cur_y_min = (np.random.rand(15,1) * im_shape[0]).astype('int')
        # cur_bboxes = np.hstack([cur_x_min, cur_y_min, \
        #     (cur_x_min + np.random.rand(15,1)*50).astype('int'), \
        #     (cur_y_min + np.random.rand(15,1)*50).astype('int')])
        # cur_confidences = np.random.rand(15)*4 - 2

        cur_bboxes = []
        cur_confidences = []
        cur_ids = []

        min_dim = min(im_shape[0], im_shape[1])
        scale_factor = 1
        threshold = -0.0 # classification threshold
        iter_ = 0
        max_iter = 100
        step_size = 1 # sliding window step size in HoG feature space
        while min_dim*scale_factor > win_size and iter_ < max_iter:
                img = cv2.resize(im, None, fx=scale_factor, fy=scale_factor)
                hog_features = vlfeat.hog.hog(img.astype(np.float64), cell_size)
                ida, idb, idc = hog_features.shape
                for j in range(0, ida-template_size, step_size):
                        for k in range(0, idb-template_size, step_size):
                                hog = hog_features[j:j+template_size, 
                                                   k:k+template_size,
                                                   :idc]
                                score = svm.decision_function(np.array([hog.ravel()]))[0]
                                if score >= threshold:
                                        cur_coordinates = np.array([k, j, k+template_size, j+template_size])*cell_size/scale_factor
                                        cur_coordinates = cur_coordinates.astype(int)
                                        cur_bboxes.append(cur_coordinates) 
                                        cur_confidences.append(score)  # keep positive
                
                scale_factor *= 0.85
                iter_ += 1

        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################

        ### non-maximum suppression ###
        # non_max_supr_bbox() can actually get somewhat slow with thousands of
        # initial detections. You could pre-filter the detections by confidence,
        # e.g. a detection with confidence -1.1 will probably never be
        # meaningful. You probably _don't_ want to threshold at 0.0, though. You
        # can get higher recall with a lower threshold. You should not modify
        # anything in non_max_supr_bbox(). If you want to try your own NMS methods,
        # please create another function.

        cur_bboxes = np.array(cur_bboxes, dtype=np.int16)
        cur_confidences = np.array(cur_confidences, dtype=np.float32).ravel()
        idsort = np.argsort(-cur_confidences)[:topk]

        cur_bboxes = cur_bboxes[idsort]
        cur_confidences = cur_confidences[idsort]
        if len(cur_bboxes) == 0:
                cur_bboxes = np.array([[0, 0, 0, 0]], dtype=np.int16)
                cur_confidences = np.array([0.])

        is_valid_bbox = non_max_suppression_bbox(cur_bboxes, cur_confidences,
            im_shape, verbose=verbose)

        print('NMS done, {:d} detections passed'.format(sum(is_valid_bbox)))
        cur_bboxes = cur_bboxes[is_valid_bbox]
        cur_confidences = cur_confidences[is_valid_bbox]

        bboxes = np.vstack((bboxes, cur_bboxes))
        confidences = np.hstack((confidences, cur_confidences))
        image_ids.extend([im_id] * len(cur_confidences))

    return bboxes, confidences, image_ids
