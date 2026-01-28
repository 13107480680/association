import numpy as np
import cv2
import torch

def get_centers_from_boxes(boxes):
    """
    从 boxes 获取中心点像素坐标。
    输入 boxes: [[x_center, y_center, w, h], ...]（归一化）
    输出: numpy array of shape (N, 2)
    """
    centers = []
    for box in boxes:
        x, y, w, h = box[2], box[3], box[4], box[5]  # 归一化的 xywh
        # 假设图像大小为 1x1（归一化坐标），稍后可以乘原图大小（可选）
        centers.append([x, y])
    return np.array(centers, dtype=np.float32)

def filt(match, boxes0, boxes1,ransacReprojThreshold=10):
    """
    过滤错误匹配并返回 refined match 和 单应矩阵 H
    """
    # 提取匹配对的中心点（像素坐标）
    pts1 = []
    pts2 = []
    for i, j in match:
        box1 = boxes0[i]
        box2 = boxes1[j]
        x1 = box1[2]
        y1 = box1[3] 
        x2 = box2[2] 
        y2 = box2[3] 
        pts1.append([x1, y1])
        pts2.append([x2, y2])
    
    pts1 = np.array(pts1, dtype=np.float32)
    pts2 = np.array(pts2, dtype=np.float32)

    # 用RANSAC估计单应矩阵
    H, mask = cv2.findHomography(pts2, pts1, method=cv2.RANSAC, ransacReprojThreshold=ransacReprojThreshold)
    if H is None:
        print("Homography estimation failed.")
        return [], None
    
    inlier_mask = mask.ravel() == 1
    match_inliers = [match[i].tolist() for i in range(len(match)) if inlier_mask[i]]
    return match_inliers, torch.tensor(H, dtype=torch.float32)

def H_youhua(match_filt, H, boxes0, boxes1, dist_thresh):
    """
    使用单应矩阵 H 扩展初始 match_filt 集合，返回新的匹配对。
    若某个框（无论图 A 或图 B）已出现在 match_filt 中，则跳过它。
    并对未匹配上的候选点进行二次筛选：若其靠近已使用的图A框，则剔除。
    """
    boxes1_centers = get_centers_from_boxes(boxes1)
    boxes1_centers = np.array(boxes1_centers, dtype=np.float32).reshape(-1, 1, 2)

    if isinstance(H, torch.Tensor):
        H = H.detach().cpu().numpy().astype(np.float32)

    boxes1_warped = cv2.perspectiveTransform(boxes1_centers, H).reshape(-1, 2)

    boxes0_centers = get_centers_from_boxes(boxes0)
    boxes0_centers = np.array(boxes0_centers, dtype=np.float32)

    match_new = match_filt.copy()
    match_new_set = set((i, j) for i, j in match_filt)  # 当前未使用，可选保留
    # 提取所有出现在匹配中的 index（无论是图 A 还是图 B）
    used_idx_a = set(i for i, _ in match_filt)
    used_idx_b = set(j for _, j in match_filt)


    buchong_id = []       # 记录补充失败的 boxes1 中 idx
    buchong_bbox = []     # 记录其映射到图 A 坐标系下的中心位置

    for idx_b, pt_b in enumerate(boxes1_warped):
        if idx_b in used_idx_b:
            continue  # 图 B 中该框已匹配

        matched = False
        for idx_a, pt_a in enumerate(boxes0_centers):
            if idx_a in used_idx_a:
                continue  # 图 A 中该框已匹配

            dist = np.linalg.norm(pt_b - pt_a)
            if dist < dist_thresh:
                match_new.append([idx_a, idx_b])
                used_idx_a.add(idx_a)
                used_idx_b.add(idx_b)
                matched = True
                break  # 每个B只匹配一次

        if not matched:
            buchong_id.append(idx_b)
            buchong_bbox.append(pt_b.tolist())

    # === 二次筛选：剔除那些靠近 already-used 图A框 的 buchong_bbox ===
    final_buchong_id = []
    final_buchong_bbox = []

    for idx_b, pt_b in zip(buchong_id, buchong_bbox):
        pt_b = np.array(pt_b)
        too_close = False
        for idx_a in used_idx_a:
            pt_a = boxes0_centers[idx_a]
            if np.linalg.norm(pt_b - pt_a) < dist_thresh:
                too_close = True
                break
        if not too_close:
            final_buchong_id.append(idx_b)
            final_buchong_bbox.append(pt_b.tolist())

    return match_new, final_buchong_id, final_buchong_bbox

# def H_youhua(match_filt, H, boxes0, boxes1, dist_thresh):
#     """
#     使用单应矩阵 H 扩展初始 match_filt 集合，返回新的匹配对。
#     若某个框（无论图 A 或图 B）已出现在 match_filt 中，则跳过它。
#     """
#     boxes1_centers = get_centers_from_boxes(boxes1)
#     boxes1_centers = np.array(boxes1_centers, dtype=np.float32).reshape(-1, 1, 2)

#     if isinstance(H, torch.Tensor):
#         H = H.detach().cpu().numpy().astype(np.float32)

#     boxes1_warped = cv2.perspectiveTransform(boxes1_centers, H).reshape(-1, 2)

#     boxes0_centers = get_centers_from_boxes(boxes0)
#     boxes0_centers = np.array(boxes0_centers, dtype=np.float32)

#     match_new = match_filt.copy()
#     match_new_set = set((i, j) for i, j in match_filt)

#     # ✅ 提取所有出现在匹配中的 index（无论是图 A 还是图 B）
#     used_idx_a = set(i for i, _ in match_filt)
#     used_idx_b = set(j for _, j in match_filt)

#     buchong_id = []       # 记录补充失败的 boxes1 中 idx
#     buchong_bbox = []  

#     for idx_b, pt_b in enumerate(boxes1_warped):
#         if idx_b in used_idx_b:
#             continue  # 图 B 中该框已匹配

#         matched = False
#         for idx_a, pt_a in enumerate(boxes0_centers):
#             if idx_a in used_idx_a:
#                 continue  # 图 A 中该框已匹配

#             dist = np.linalg.norm(pt_b - pt_a)
#             if dist < dist_thresh :
#                 match_new.append([idx_a, idx_b])
#                 match_new_set.add((idx_a, idx_b))
#                 used_idx_a.add(idx_a)
#                 used_idx_b.add(idx_b)
#                 matched = True               
#                 break  # 每个B只匹配一次
#         if not matched:
#             # 虽未匹配成功，记录映射后仍在图 A 范围内的目标框中心
#             buchong_id.append(idx_b)
#             buchong_bbox.append(pt_b.tolist())  # 加入映射后的中心点（图 A 坐标）

#     return match_new, buchong_id, buchong_bbox         

