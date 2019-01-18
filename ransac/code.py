import numpy as np


def calculate_projection_matrix(points_2d, points_3d):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points:

                                                      [ M11      [ u1
                                                        M12        v1
                                                        M13        .
                                                        M14        .
    [ X1 Y1 Z1 1 0  0  0  0 -u1*X1 -u1*Y1 -u1*Z1        M21        .
      0  0  0  0 X1 Y1 Z1 1 -v1*X1 -v1*Y1 -v1*Z1        M22        .
      .  .  .  . .  .  .  .    .     .      .       *   M23   =    .
      Xn Yn Zn 1 0  0  0  0 -un*Xn -un*Yn -un*Zn        M24        .
      0  0  0  0 Xn Yn Zn 1 -vn*Xn -vn*Yn -vn*Zn ]      M31        .
                                                        M32        un
                                                        M33 ]      vn ]

    Then you can solve this using least squares with np.linalg.lstsq() or SVD.
    Notice you obtain 2 equations for each corresponding 2D and 3D point
    pair. To solve this, you need at least 6 point pairs.

    Args:
    -   points_2d: A numpy array of shape (N, 2)
    -   points_2d: A numpy array of shape (N, 3)

    Returns:
    -   M: A numpy array of shape (3, 4) representing the projection matrix
    """

    # Placeholder M matrix. It leads to a high residual. Your total residual
    # should be less than 1.
    M = np.asarray([[0.1768, 0.7018, 0.7948, 0.4613],
                    [0.6750, 0.3152, 0.1136, 0.0480],
                    [0.1020, 0.1725, 0.7244, 0.9932]])

    ###########################################################################
    # TODO: YOUR PROJECTION MATRIX CALCULATION CODE HERE
    ###########################################################################

    # points_2d = (ui, vi)
    # points_3d = (Xi, Yi, Zi)
    A = np.zeros((points_2d.shape[0]*2, 11))
    for i in range(0, A.shape[0], 2):
        j = i // 2
        
        # u
        A[i][0] = points_3d[j][0] # X1
        A[i][1] = points_3d[j][1] # Y1
        A[i][2] = points_3d[j][2] # Z1
        A[i][3] = 1
        A[i][4:8] = 0
        A[i][8] = -points_2d[j][0]*points_3d[j][0]
        A[i][9] = -points_2d[j][0]*points_3d[j][1]
        A[i][10] = -points_2d[j][0]*points_3d[j][2]
        
        # v
        A[i+1][:4] = 0
        A[i+1][4] = points_3d[j][0]
        A[i+1][5] = points_3d[j][1]
        A[i+1][6] = points_3d[j][2]
        A[i+1][7] = 1
        A[i+1][8] = -points_2d[j][1]*points_3d[j][0]
        A[i+1][9] = -points_2d[j][1]*points_3d[j][1]
        A[i+1][10] = -points_2d[j][1]*points_3d[j][2] 
    right = points_2d.flatten()
    evM = np.linalg.lstsq(A, right, rcond=None)   
    result = np.append(evM[0], 1)
    result = result.reshape(3, 4)
    M = result

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return M

def calculate_camera_center(M):
    """
    Returns the camera center matrix for a given projection matrix.

    The center of the camera C can be found by:

        C = -Q^(-1)m4

    where your project matrix M = (Q | m4).

    Args:
    -   M: A numpy array of shape (3, 4) representing the projection matrix

    Returns:
    -   cc: A numpy array of shape (1, 3) representing the camera center
            location in world coordinates
    """

    # Placeholder camera center. In the visualization, you will see this camera
    # location is clearly incorrect, placing it in the center of the room where
    # it would not see all of the points.
    cc = np.asarray([1, 1, 1])

    ###########################################################################
    # TODO: YOUR CAMERA CENTER CALCULATION CODE HERE
    ###########################################################################

    cc = -np.linalg.inv(M[:, :3]).dot(M[:, 3])

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return cc

# def normalise2dpts(pts):

#     if pts.shape[0] != 3:
#         print("Input shoud be 3")

#     finiteind = np.nonzero(abs(pts[2,:]) > np.spacing(1));
 
#     dist = []
#     for i in finiteind:
#         pts[0,i] = pts[0,i]/pts[2,i]
#         pts[1,i] = pts[1,i]/pts[2,i]
#         pts[2,i] = 1;
#         c = np.mean(pts[0:2,i].T, axis=0).T          
#         newp1 = pts[0,i]-c[0]
#         newp2 = pts[1,i]-c[1]
#         dist.append(np.sqrt(newp1**2 + newp2**2))
        
#     meandist = np.mean(dist[:])
#     scale = np.sqrt(2)/meandist
#     T = np.array([[scale, 0, -scale*c[0]], [0, scale, -scale*c[1]], [0, 0, 1]])
#     newpts = T.dot(pts)

#     return [newpts, T]

# def estimate_fundamental_matrix(points_a, points_b):
#     """
#     Calculates the fundamental matrix. Try to implement this function as
#     efficiently as possible. It will be called repeatedly in part 3.

#     You must normalize your coordinates through linear transformations as
#     described on the project webpage before you compute the fundamental
#     matrix.

#     Args:
#     -   points_a: A numpy array of shape (N, 2) representing the 2D points in
#                   image A
#     -   points_b: A numpy array of shape (N, 2) representing the 2D points in
#                   image B

#     Returns:
#     -   F: A numpy array of shape (3, 3) representing the fundamental matrix
#     """

#     # Placeholder fundamental matrix
#     F = np.asarray([[0, 0, -0.0004],
#                     [0, 0, 0.0032],
#                     [0, -0.0044, 0.1034]])

#     ###########################################################################
#     # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
#     ###########################################################################
#     num = points_a.shape[0]

#     points_a = np.hstack([points_a, np.ones((points_a.shape[0], 1))])
#     points_b = np.hstack([points_b, np.ones((points_b.shape[0], 1))])
#     x1, Ta = normalise2dpts(points_a.T)
#     x2, Tb = normalise2dpts(points_b.T)
#     A = np.ones((num, 9))
#     for i in range(num):
#         u1 = x1[0, i]
#         v1 = x1[1, i]
#         u2 = x2[0, i]
#         v2 = x2[1, i]
#         A[i, :8] = [u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1]

#     _, _, V = np.linalg.svd(A)
#     F = V[-1].reshape(3,3)
    
#     (U, S, V) = np.linalg.svd(F)
#     S[2] = 0
#     F = U.dot(np.diag(S)).dot(V)
#     F = F/F[2,2]

#     ###########################################################################
#     # END OF YOUR CODE
#     ###########################################################################

#     return Tb.T.dot(F).dot(Ta)

def estimate_fundamental_matrix(points_a, points_b):
    """
    Calculates the fundamental matrix. Try to implement this function as
    efficiently as possible. It will be called repeatedly in part 3.

    You must normalize your coordinates through linear transformations as
    described on the project webpage before you compute the fundamental
    matrix.

    Args:
    -   points_a: A numpy array of shape (N, 2) representing the 2D points in
                  image A
    -   points_b: A numpy array of shape (N, 2) representing the 2D points in
                  image B

    Returns:
    -   F: A numpy array of shape (3, 3) representing the fundamental matrix
    """
    ###########################################################################
    # TODO: YOUR FUNDAMENTAL MATRIX ESTIMATION CODE HERE
    ###########################################################################
    num = points_a.shape[0]
    A = np.zeros((num, 8))
    ones = np.ones((num, 1))
    
    cu_a = np.mean(points_a[:, 0])
    cv_a = np.mean(points_a[:, 1])
    s_a = np.std(points_a - np.mean(points_a))
    Ta = np.mat([[1/s_a, 0, 0], [0, 1/s_a, 0], [0, 0, 1]]) * np.mat([[1, 0, -cu_a], [0,1,-cv_a], [0, 0, 1]])
    Ta = np.array(Ta)
    points_a = np.hstack([points_a, ones])
    
    cu_b = np.mean(points_a[:, 0])
    cv_b = np.mean(points_a[:, 1])
    s_b = np.std(points_a - np.mean(points_a))
    Tb = np.mat([[1/s_b, 0, 0], [0, 1/s_b, 0], [0, 0, 1]]) * np.mat([[1, 0, -cu_b], [0,1,-cv_b], [0, 0, 1]])
    Tb = np.array(Tb)
    points_b = np.hstack([points_b, ones])
    
    for i in range(num):
        points_a[i] = np.matmul(Ta, points_a[i])
        points_b[i] = np.matmul(Tb, points_b[i])
        
        u1 = points_a[i, 0]
        v1 = points_a[i, 1]
        u2 = points_b[i, 0]
        v2 = points_b[i, 1]
        A[i] = [u1*u2, v1*u2, u2, u1*v2, v1*v2, v2, u1, v1]
    
    A = np.hstack([A, ones])
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3,3)
    
    (U, S, V) = np.linalg.svd(F)
    S[2] = 0
    F = U.dot(np.diag(S)).dot(V)
    F = F/F[2,2]
    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################
    return Tb.T.dot(F).dot(Ta)

def ransac_fundamental_matrix(matches_a, matches_b):
    # Placeholder values
    best_F = estimate_fundamental_matrix(matches_a[:10, :], matches_b[:10, :])
    inliers_a = matches_a[:100, :]
    inliers_b = matches_b[:100, :]

    ###########################################################################
    # TODO: YOUR RANSAC CODE HERE
    ###########################################################################
    k = 2000 # maximum iterations allowed in the algorithm
    n = 10 # the minimum number of data values required to fit the model
    threshold = 0.05 # the number of close data values required to assert that a model fits well to data
    iters = 0
    # bestF
    besterr = np.inf
    bestinlier = []
    
    # 3. threshold on how many points is inlier      
    while iters < k:
        inliers = []
        # 1. sample n rows
        index = [np.random.randint(low = 0, high = len(matches_a)) for i in range(n)]
        sample_a = matches_a[index]
        sample_b = matches_b[index]
        # 2. calculate the F
        F = estimate_fundamental_matrix(sample_a, sample_b)
        for i in range(len(matches_a)):
        # calculate the distance between each transformed point
            x = np.hstack([matches_a[i], 1])
            x_= np.hstack([matches_b[i], 1])
            dis_mat = np.abs(x.dot(F.T).dot(x_.T)) # need abs?
            if dis_mat < threshold:
                inliers.append([dis_mat, matches_a[i], matches_b[i]])
        
        if len(inliers) > len(bestinlier):
            bestinlier = inliers
            bestF = F
        iters += 1
    
    for i in range(min(100, len(bestinlier))):
        inliers_a[i, :] = bestinlier[i][1]
        inliers_b[i, :] = bestinlier[i][2]

    ###########################################################################
    # END OF YOUR CODE
    ###########################################################################

    return best_F, inliers_a, inliers_b