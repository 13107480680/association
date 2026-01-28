import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from zz.all_nn.backbone_match import TargetConnectionModel
from zz.test_method.caculate_folder import test_folder
from zz.all_nn.RKHS_filt import RKHS_filter_Model_test,RKHS_filter_Model1,RKHS_filter_Model_jian


class ImageMatcher:
    def __init__(self, model_path=None,filter_model_path=None,backbone_out_dim=None, filter_threshold=0.00001,lightglue_pretrain=None):
        # if superpoint_config is None:
        #     superpoint_config = {"weights": "weights/superpoint_lightglue", "input_dim": 2048}

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TargetConnectionModel(
            backbone_out_dim=backbone_out_dim,
            filter_threshold=filter_threshold,
            lightglue_pretrain=lightglue_pretrain
        ).to(self.device)##########################################################改这里就可以了，不同的模型
        # 加载预训练权重
        if os.path.exists(model_path):
            print(f"Loading pretrained model from {model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=True)
        else:
            raise FileNotFoundError(f"Pretrained model not found at {model_path}")
        
        self.filter_model=RKHS_filter_Model_test(dim=512,att_layer=4).to(self.device)
        if os.path.exists(filter_model_path):
            print(f"Loading pretrained model from {filter_model_path}")
            checkpoint = torch.load(filter_model_path, map_location=self.device)
            self.filter_model.load_state_dict(checkpoint, strict=True)
        else:
            raise FileNotFoundError(f"Pretrained model not found at {filter_model_path}")        


        # 切换为评估模式
        self.model.eval()


if __name__ == "__main__":
    
    img_folder_a = r"/mnt/16T/data/zz/zz/code/new_t/img_a"
    txt_folder_a = r"/mnt/16T/data/zz/zz/code/new_t/txt_a"
    img_folder_b = r"/mnt/16T/data/zz/zz/code/new_t/img_b"
    txt_folder_b = r"/mnt/16T/data/zz/zz/code/new_t/txt_b"

    # img_folder_a = r"/mnt/16T/data/zz/zz/data/test/img_a"
    # txt_folder_a = r"/mnt/16T/data/zz/zz/data/test/txt_a"
    # img_folder_b = r"/mnt/16T/data/zz/zz/data/test/img_b"
    # txt_folder_b = r"/mnt/16T/data/zz/zz/data/test/txt_b"

    # img_folder_a=r"/mnt/16T/data/zz/zz/data/tianda_zhengshi_gap10/train/img_a"
    # txt_folder_a=r"/mnt/16T/data/zz/zz/data/tianda_zhengshi_gap10/train/txt_a"
    # img_folder_b=r"/mnt/16T/data/zz/zz/data/tianda_zhengshi_gap10/train/img_b"
    # txt_folder_b=r"/mnt/16T/data/zz/zz/data/tianda_zhengshi_gap10/train/txt_b"


    # img_folder_a = r"/mnt/8t/zz/zz/baseline_superglue_TMM/lunwen1_pic_out/attention_get/img_a"
    # txt_folder_a = r"/mnt/8t/zz/zz/baseline_superglue_TMM/lunwen1_pic_out/attention_get/txt_a"
    # img_folder_b = r"/mnt/8t/zz/zz/baseline_superglue_TMM/lunwen1_pic_out/attention_get/img_b"
    # txt_folder_b = r"/mnt/8t/zz/zz/baseline_superglue_TMM/lunwen1_pic_out/attention_get/txt_b"
    model_path_set=r"/mnt/16T/data/zz/zz/paper2/duibi_algorithm/code/paper2_github/model_20251221_035458_epoch_196.pth"
    filter_model_path=r"/mnt/16T/data/zz/zz/paper2/duibi_algorithm/code/paper2_github/model_20260110_171524_epoch_40.pth"
    # /mnt/8t/zz/zzbaseline_superglue_TMM/lunwen1_pic_out/MDMT_association/proposed/model_20250905_154515_epoch_20.pth
    # /mnt/8t/zz/zz/code/mutitarget_connection/0902beifeng_superglue_gmm/out_pth/gap10_yuanban_0905/model_20250905_161043_epoch_26.pth
    # /mnt/8t/zz/zz/code/mutitarget_connection/0409chushi/fastrcnn_LightGlue/out_pth/small_tianda0408_roi/2048origion/model_20250410_135842_epoch_32.pth
    # img_folder_a = r"/mnt/8t/zz/zz/data/stark3/gap20_track3_copy/test/img_a"
    # txt_folder_a = r"/mnt/8t/zz/zz/data/stark3/gap20_track3_copy/test/txt_a"
    # img_folder_b = r"/mnt/8t/zz/zz/data/stark3/gap20_track3_copy/test/img_b"
    # txt_folder_b = r"/mnt/8t/zz/zz/data/stark3/gap20_track3_copy/test/txt_b"
    ###############
    ##可见光#######
    ##############
    # txt_folder_a = r"/mnt/8t/zz/zz/data/stark3/gap10_track3/test/txt_a"
    # txt_folder_b  = r"/mnt/8t/zz/zz/data/stark3/gap10_track3/test/txt_b"
    # img_folder_a  = r"/mnt/8t/zz/zz/data/stark3/gap10_track3/test/img_a"
    # img_folder_b = r"/mnt/8t/zz/zz/data/stark3/gap10_track3/test/img_b"
    # model_path_set=r"/mnt/8t/zz/zz/baseline_superglue_TMM/out_pth/0922track3bt_wuyuxunlian_new/model_20250922_213551_epoch_16.pth"
    # 
    ###############
    ##红外#######
    ##############
    # txt_folder_a = r"/mnt/8t/zz/zz/data/stark3/gap10_hongwai/test/txt_a"
    # txt_folder_b  = r"/mnt/8t/zz/zz/data/stark3/gap10_hongwai/test/txt_b"
    # img_folder_a  = r"/mnt/8t/zz/zz/data/stark3/gap10_hongwai/test/img_a"
    # img_folder_b = r"/mnt/8t/zz/zz/data/stark3/gap10_hongwai/test/img_b"
    # model_path_set=r"/mnt/8t/zz/zz/baseline_superglue_TMM/out_pth/0922track3bt8_hongwai_wuyuxunlianquanzhong/model_20250922_194952_epoch_6.pth"


    lightglue_pretrain_path="/mnt/8t/zz/zz/code/mutitarget_connection/0409chushi/fastrcnn_LightGlue/weights/superpoint_lightglue.pth"
    filter_threshold_set=0.00000000001
    filter_threshold_set=0
    backbone_out_dim_set=2048
    vec_thr=0.8
    # 初始化匹配器
    matcher = ImageMatcher(model_path=model_path_set,
                           filter_model_path=filter_model_path,
                           backbone_out_dim=backbone_out_dim_set,
                           filter_threshold=filter_threshold_set,
                           lightglue_pretrain=None)
    


    #画图地址
    if_plot_match=0
    output_folder = r"/mnt/16T/data/zz/zz/paper2/duibi_algorithm/out_pic/MDMT/reid"

    if_att0_pic=0#开关
    save_attn_path0=r'/mnt/16T/data/zz/zz/code/paper2_new1213/1211base_superglue_papper1_train_baohu/lunwen1_pic_out/attention_get/att0'
    idx_attn_0=0

    if_att1_pic=0
    save_attn_path1=r'/mnt/16T/data/zz/zz/code/paper2_new1213/1211base_superglue_papper1_train_baohu/lunwen1_pic_out/attention_get/att1'
    idx_attn_1=0

    if_cross_pic=0
    save_path_cross=r'/mnt/16T/data/zz/zz/code/paper2_new1213/1211base_superglue_papper1_train_baohu/lunwen1_pic_out/attention_get/corss'
    idx_cross_attn=0
    # 执行测试

    test_folder(matcher.device,matcher.model,matcher.filter_model,
                img_folder_a, txt_folder_a, img_folder_b, txt_folder_b, output_folder,
                save_attn_path0,idx_attn_0,save_attn_path1,idx_attn_1,save_path_cross,idx_cross_attn,
                  if_att0_pic,if_att1_pic,if_cross_pic,if_plot_match,vec_thr=vec_thr)










    



