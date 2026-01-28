import cv2
import numpy as np

def estimate_homography_from_boxes(boxes0, boxes1, match_filt):
    """
    估计从图2到图1的单应性矩阵 H (3x3).
    
    Args:
        boxes0 (list): 图1的检测框, 每个元素为 [类别, di, x, y, w, h]，其中 (x, y) 是中心
        boxes1 (list): 图2的检测框, 每个元素为 [类别, di, x, y, w, h]，其中 (x, y) 是中心
        match_filt (list): 匹配对 [[i0, j0], [i1, j1], ...],
                           表示 boxes0[i0] 与 boxes1[j0] 是一对匹配
    
    Returns:
        H (np.ndarray or None): 3x3 单应性矩阵 (图2 → 图1)，若失败则返回 None
    """
    if len(match_filt) < 4:
        print("Not enough matches for homography estimation (need at least 4).")
        return None

    src_pts = []  # 图2中的点 (源) → boxes1
    dst_pts = []  # 图1中的点 (目标) → boxes0

    for i0, j1 in match_filt:
        # 图1: boxes0[i0]
        x0, y0 = boxes0[i0][2], boxes0[i0][3]  # 中心坐标
        
        # 图2: boxes1[j1]
        x1, y1 = boxes1[j1][2], boxes1[j1][3]  # 中心坐标

        src_pts.append([x1, y1])  # 源：图2
        dst_pts.append([x0, y0])  # 目标：图1

    src_pts = np.float32(src_pts)  # shape: (N, 2)
    dst_pts = np.float32(dst_pts)  # shape: (N, 2)

    # 使用 RANSAC 估计 H（图2 → 图1）
    H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=10)

    if H is None or np.sum(mask) < 4:
        print(f"Homography estimation failed: only {np.sum(mask)} inliers.")
        return None

    return H