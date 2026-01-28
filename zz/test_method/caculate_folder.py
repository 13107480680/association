import os
import time
import re
import torch
from zz.test_method.score import single_scores
from zz.test_method.process_pair import process_pair
from zz.filter.ransac_filt import H_youhua,filt
from zz.plot.test_plot import plot_yinshe_zuoyou,plot_yinshe_shangxia
from zz.filter.guss_filt import fg_gmm_rigid_match
from zz.all_nn.RKHS_filt import RKHS_filter_Model
from zz.filter.get_H import estimate_homography_from_boxes

# def extract_number(filename):
#     # 使用正则表达式匹配文件名中的数字部分
#     match = re.search(r'\d+', filename)  # 匹配第一个连续的数字
#     return int(match.group()) if match else float('inf')
def extract_number(filename):
    # 匹配文件名中所有数字
    numbers = re.findall(r'\d+', filename)
    # 取最后一个数字作为排序依据
    return int(numbers[-1]) if numbers else float('inf')

def get_match_id_mask(boxes0, boxes1, match):
    match_mask = []
    for idx0, idx1 in match.tolist():
        id0 = boxes0[idx0][1]  # 目标id在第2列，即索引1
        id1 = boxes1[idx1][1]
        match_mask.append(id0 == id1)
    return match_mask

def test_folder(device, model,RKHS_filter_Model,
                img_folder_a, txt_folder_a, img_folder_b, txt_folder_b, output_folder,
                save_attn_path0,idx_attn_0,save_attn_path1,idx_attn_1,save_path_cross,idx_cross_attn,
                if_att0_pic,if_att1_pic,if_cross_pic,if_plot_match,vec_thr=None):
    os.makedirs(output_folder, exist_ok=True)
    # 按照文件名中的数字排序
    img_files_a = sorted([f for f in os.listdir(img_folder_a) if f.endswith((".jpg", ".png"))],key=extract_number)
    txt_files_a = sorted([f for f in os.listdir(txt_folder_a) if f.endswith(".txt")],key=extract_number)
    img_files_b = sorted([f for f in os.listdir(img_folder_b) if f.endswith((".jpg", ".png"))],key=extract_number)
    txt_files_b = sorted([f for f in os.listdir(txt_folder_b) if f.endswith(".txt")],key=extract_number)
    # 确保文件数量一致
    assert len(img_files_a) == len(txt_files_a), "Mismatch between images and texts in folder A"
    assert len(img_files_b) == len(txt_files_b), "Mismatch between images and texts in folder B"
    assert len(img_files_a) == len(img_files_b), "Mismatch between folder A and folder B"


    ######################
    ##存放指标的容器####
    ######################    
    time_each=[]    
    num_pict=0
    test_scores=[]
    test_precision=[]
    test_recall=[]

    time_each_new=[]
    test_scores_new=[]
    test_precision_new=[]
    test_recall_new=[]
   

    sim_before=[]
    sim_after=[]
    ##计算F1曲线
    f1=[]
    f_presion=[]
    f_recall=[]

    ##########################
    ##开始循环遍历整个数据集####
    ##########################   


    for img_file_a, txt_file_a, img_file_b, txt_file_b in zip(img_files_a, txt_files_a, img_files_b, txt_files_b):
        print(f"Processing: {img_file_a} & {img_file_b}")
        img_path_a = os.path.join(img_folder_a, img_file_a)
        img_path_b = os.path.join(img_folder_b, img_file_b)
        txt_path_a = os.path.join(txt_folder_a, txt_file_a)
        txt_path_b = os.path.join(txt_folder_b, txt_file_b)


        ######################
        ##算法推理的核心步骤####
        ######################
        start_time=time.time()
        
        boxes0 ,boxes1,matches, features_a, features_b,image0,image1,scores,self_attn0,self_attn1,cross_attn01,cross_attn10,out_desc0,out_desc1,padded_keypoints_a,padded_keypoints_b=process_pair(device,model,img_path_a, txt_path_a, img_path_b, txt_path_b)

        # boxes0 ,boxes1,matches, features_a, features_b,image0,image1,scores,self_attn0,self_attn1,cross_attn01,cross_attn10,lap_pe0,lap_pe1=process_pair(img_path_a, txt_path_a, img_path_b, txt_path_b, output_path)
        end_time=time.time()

        time_each.append((end_time-start_time)*1000)
        num_pict+=1
        print(f"算法的运推理时间：{(end_time-start_time)*1000}ms")
        dd0=(end_time-start_time)*1000
        features_1,features_2,match,score=features_a[0].tolist(),features_b [0].tolist(),matches[0],scores[0].tolist()
        #############
        ##错误过滤####
        #############
        match00=match.tolist()
        ###########
        ##方法1RHSK
        ############
        # match_RKHS=(boxes0 ,boxes1,fusion_features_a,fusion_features_b,match00,scores)
        ###########
        ##方法2ransac
        ############
        # if match.shape[0]<6: 
        #     match_filt=match00
        #     match_new=match00
        # else :
        vector_score,f_fusion,f_origin,f_global=RKHS_filter_Model(out_desc0,out_desc1,matches[0].unsqueeze(0), padded_keypoints_a.float(),padded_keypoints_b.float())
        mask = vector_score.squeeze(0) > vec_thr
        match_filt = matches[0][mask].tolist()
        H=estimate_homography_from_boxes(boxes0, boxes1, match_filt)
        if H is not None:
            dd=1
            # match_filt, H, =filt(match, boxes0, boxes1,ransacReprojThreshold=20)#Ransac#10是默认
            # _, H, =filt(match, boxes0, boxes1,ransacReprojThreshold=10)#使用高斯混合模型
            # match_filt, P_final =fg_gmm_rigid_match(boxes0, boxes1,  match00, features_1, features_2, score, max_iter=4, tol=1e-4, alpha=1)
            match_new,buchong_id, buchong_bbox=H_youhua(match_filt, H,  boxes0, boxes1, dist_thresh=70)#过滤后，估计H用于映射  #buchong_id, buchong_bbox是遮挡隐射到图a的坐标#80是默认
            img_h, img_w = image0.shape[1], image0.shape[2]
            filtered_bboxes = []#这是映射过去没有适配的，也就是抗遮挡的
            filtered_ids = []
            for center, obj_id in zip(buchong_bbox, buchong_id):
                x, y = center
                if 0 <= x < img_w and 0 <= y < img_h:
                    filtered_bboxes.append(center)
                    filtered_ids.append(obj_id)
            end_time=time.time()
        else:
            match_filt=match00
            match_new=match00
            filtered_bboxes = []#这是映射过去没有适配的，也就是抗遮挡的
            filtered_ids = []            
        print(f"过滤后算法的运推理时间：{(end_time-start_time)*1000}ms")
        dd=(end_time-start_time)*1000
        time_each_new.append((end_time-start_time)*1000)

        ##############
        ##计算单帧指标##
        ##############
        single_score,single_precision,single_recall=single_scores(boxes0 ,boxes1,matches[0])
        single_score_new,single_precision_new,single_recall_new=single_scores(boxes0 ,boxes1,match_new)
        # print(f"过滤前:{single_score}")
        # print(f"过滤后:{single_score_new}")
        test_scores.append(single_score)
        test_precision.append(single_precision)
        test_recall.append(single_recall)

        test_scores_new.append(single_score_new)
        test_precision_new.append(single_precision_new)
        test_recall_new.append(single_recall_new)

        ###########
        ##画图####
        ###########
        match_mask=get_match_id_mask(boxes0, boxes1, matches[0])
        if if_plot_match==1:
            # plot_yinshe_zuoyou(match00, match_filt, match_new, image0, image1, boxes0, boxes1,output_folder,img_file_a)
            dd=dd+10####################
            # plot_yinshe_zuoyou(match00, match_filt, match_new, image0, image1, boxes0, boxes1,output_folder,img_file_a,single_score_new,single_precision_new,single_recall_new,dd)
            plot_yinshe_zuoyou(match00, match_filt, match_new, image0, image1, boxes0, boxes1,output_folder,img_file_a,single_score_new,single_precision_new,single_recall_new,dd,filtered_ids,filtered_bboxes)
 



    #################
    ##计算所有帧指标##
    #################
    dd=1
