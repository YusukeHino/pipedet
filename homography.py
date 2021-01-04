import numpy as np
import cv2
# from matplotlib import pyplot as plt

def abs_xywh_to_rel_xyxy(bbox, original_size):
    w, h = original_size
    x1, y1, w_abs, h_abs = bbox
    x2 = x1 + w_abs
    y2 = y1 + h_abs
    return [x1/w, y1/h, x2/w, y2/h]

# _path_former_frame = '/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_024/mirror_2/frames/0001.png'
# _original_size_former_frame = (89, 117)
# _bbox_former_frame = [63.6,26.74,16.190000000000005,28.790000000000003]
# _rel_bbox_former_frame = abs_xywh_to_rel_xyxy(_bbox_former_frame, _original_size_former_frame)

# _path_latter_frame = '/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_024/mirror_2/frames/0031.png'
# _original_size_latter_frame = (101, 138)
# _bbox_latter_frame = [72.7,34.65,16.28,35.550000000000004]
# _rel_bbox_latter_frame = abs_xywh_to_rel_xyxy(_bbox_latter_frame, _original_size_latter_frame)


_path_former_frame = '/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_024/mirror_2/frames/0041.png'
_original_size_former_frame = (109, 145)
_bbox_former_frame = [75.1,39.37,17.200000000000003,37.160000000000004]
_rel_bbox_former_frame = abs_xywh_to_rel_xyxy(_bbox_former_frame, _original_size_former_frame)

_path_latter_frame = '/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_024/mirror_2/frames/0071.png'
_original_size_latter_frame = (141, 192)
_bbox_latter_frame = [90.0,66.71,21.200000000000003,48.43000000000001]
_rel_bbox_latter_frame = abs_xywh_to_rel_xyxy(_bbox_latter_frame, _original_size_latter_frame)


# _path_former_frame = '/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_024/mirror_2/frames/0070.png'
# _original_size_former_frame = (140, 189)
# _bbox_former_frame = [88.68,65.87,21.50999999999999,48.42999999999999]
# _rel_bbox_former_frame = abs_xywh_to_rel_xyxy(_bbox_former_frame, _original_size_former_frame)

# _path_latter_frame = '/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_024/mirror_2/frames/0105.png'
# _original_size_latter_frame = (174, 242)
# _bbox_latter_frame = [97.7,108.52,27.33,60.13000000000001]
# _rel_bbox_latter_frame = abs_xywh_to_rel_xyxy(_bbox_latter_frame, _original_size_latter_frame)


# _path_former_frame = '/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_024/mirror_2/frames/0070.png'
# _original_size_former_frame = (140, 189)
# _bbox_former_frame = [88.68,65.87,21.50999999999999,48.42999999999999]
# _rel_bbox_former_frame = abs_xywh_to_rel_xyxy(_bbox_former_frame, _original_size_former_frame)

# _path_latter_frame = '/home/appuser/data/facing_via_mirror/3840_2160_30fps/mirror_seq/20200918_024/mirror_2/frames/0106.png'
# _original_size_latter_frame = (174, 246)
# _bbox_latter_frame = [97.91,110.67,27.340000000000003,62.730000000000004]
# _rel_bbox_latter_frame = abs_xywh_to_rel_xyxy(_bbox_latter_frame, _original_size_latter_frame)


def see_if_is_located_in_center(coordinate, size):
    """
    Args:
        coordinate: Tuple[int, int]
        size: Tuple[int, int]
            width, height
    Returns:
        bool
    """
    width, height = size
    x, y = coordinate
    if x <= 1/4 * width:
        return False
    if x >= 3/4 * width:
        return False
    if y <= 1/4 * height:
        return False
    if y >= 3/4 * height:
        return False
    return True

def remove_out_of_center(keypoints, descriptors, original_size):
    ret_keypoints = []
    ret_indxs_of_descriptor = np.array([], dtype=np.int8)
    ret_descriptors = np.array([])

    for cnt, (keypoint, descriptor) in enumerate(zip(keypoints, descriptors)):
        if see_if_is_located_in_center(keypoint.pt, original_size):
            ret_keypoints.append(keypoint)
            ret_indxs_of_descriptor = np.append(ret_indxs_of_descriptor, cnt)
    if len(ret_indxs_of_descriptor):
        ret_descriptors = descriptors[ret_indxs_of_descriptor]
    return ret_keypoints, ret_descriptors

def see_if_is_inside_bbox(coordinate, size, bbox):
    width, height = size
    x, y = coordinate
    x1, y1, x2, y2 = bbox
    if x/width < x1:
        return False
    if y/height < y1:
        return False
    if x/width > x2:
        return False
    if y/height > y2:
        return False
    return True

def remove_inside_bbox(keypoints, descriptors, original_size, bbox):
    ret_keypoints = []
    ret_indxs_of_descriptor = np.array([], dtype=np.int8)
    ret_descriptors = np.array([])

    for cnt, (keypoint, descriptor) in enumerate(zip(keypoints, descriptors)):
        if not see_if_is_inside_bbox(keypoint.pt, original_size, bbox):
            ret_keypoints.append(keypoint)
            ret_indxs_of_descriptor = np.append(ret_indxs_of_descriptor, cnt)
    if len(ret_indxs_of_descriptor):
        ret_descriptors = descriptors[ret_indxs_of_descriptor]
    return ret_keypoints, ret_descriptors

def remove_outside_brim_with_ellipse(keypoints, descriptors, original_size):
    ret_keypoints = []
    ret_indxs_of_descriptor = np.array([], dtype=np.int8)
    ret_descriptors = np.array([])
    
    width, height = original_size
    alpha = (1/6,)
    beta = (1/16, 1/8)
    for cnt, (keypoint, descriptor) in enumerate(zip(keypoints, descriptors)):
        x, y = keypoint.pt
        if (x - 1/2 * width) ** 2 / (1/2 * width - beta[0]*width) ** 2 + (y - 1/2 * height) ** 2 / (1/2*height-beta[1]*height) ** 2 <= 1:
            ret_keypoints.append(keypoint)
            ret_indxs_of_descriptor = np.append(ret_indxs_of_descriptor, cnt)
    if len(ret_indxs_of_descriptor):
        ret_descriptors = descriptors[ret_indxs_of_descriptor]
    return ret_keypoints, ret_descriptors


MIN_MATCH_COUNT = 10

img1 = cv2.imread(_path_former_frame,0) # queryImage
img2 = cv2.imread(_path_latter_frame,0) # trainImage

img1 = cv2.resize(img1, (512, 512))
img2 = cv2.resize(img2, (512, 512))

# Initiate SIFT detector
sift = cv2.ORB_create(nfeatures = 10000, edgeThreshold = 17, patchSize=17)

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(img1,None)
kp2, des2 = sift.detectAndCompute(img2,None)

# kp1, des1 = remove_out_of_center(kp1, des1, img1.shape)
# kp2, des2 = remove_out_of_center(kp2, des2, img2.shape)

kp1, des1 = remove_outside_brim_with_ellipse(kp1, des1, img1.shape)
kp2, des2 = remove_outside_brim_with_ellipse(kp2, des2, img2.shape)

kp1, des1 = remove_inside_bbox(kp1, des1, img1.shape, _rel_bbox_former_frame)
kp2, des2 = remove_inside_bbox(kp2, des2, img2.shape, _rel_bbox_latter_frame)

# FLANN_INDEX_KDTREE = 0
# index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
# search_params = dict(checks = 50)

# flann = cv2.FlannBasedMatcher(index_params, search_params)
bf = cv2.BFMatcher(cv2.NORM_HAMMING2, crossCheck=True)

matches = bf.match(des1, des2)

good = sorted(matches, key = lambda x:x.distance)

# # store all the good matches as per Lowe's ratio test.
# good = []
# for m,n in matches:
#     if m.distance < 0.7*n.distance:
#         good.append(m)


if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

    # original box depict
    depict_param = (2, 55, 330, 2, 2, 5) if w > 1024 else (1, 15, 66, 1, 0.4, 1)
    cv2.rectangle(
        img=img1,
        pt1=(int(_rel_bbox_former_frame[0]*w), int(_rel_bbox_former_frame[1]*h)),
        pt2=(int(_rel_bbox_former_frame[2]*w), int(_rel_bbox_former_frame[3]*h)),
        color=(0,255,0),
        thickness=3
        )
    cv2.rectangle(
        img=img2,
        pt1=(int(_rel_bbox_latter_frame[0]*w), int(_rel_bbox_latter_frame[1]*h)),
        pt2=(int(_rel_bbox_latter_frame[2]*w), int(_rel_bbox_latter_frame[3]*h)),
        color=(0,255,0),
        thickness=3
        )



    # for box transform
    x1, y1, x2, y2 = (rel*w if cnt%2==0 else rel*h for cnt,rel in enumerate(_rel_bbox_former_frame))
    pts_for_box = np.float32([ [x1,y1],[x1,y2],[x2,y2],[x2,y1] ]).reshape(-1,1,2)
    dst_for_box = cv2.perspectiveTransform(pts_for_box,M)

    img2 = cv2.polylines(img2,[np.int32(dst_for_box)],True,255,3, cv2.LINE_AA)
else:
    print("Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT))
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = cv2.DrawMatchesFlags_DEFAULT)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

# plt.imshow(img3, 'gray'),plt.show()

cv2.imwrite('/home/appuser/src/pipedet/tests/demo_for_lumix/ro_tracking_result/board/matching.png',img3)