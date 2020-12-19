_base_ = "./ggnn_faster_rcnn_r50_fpn_sample1e-3_mstrain_1x_vg1000_random_hierarcal.py"
model = dict(
    roi_head=dict(
        bbox_head=dict(
            ggnn_config=dict(
                adjecent_path="./data/vg/adjecent_attr.pt",
                initweight_path="./data/vg/init_weights_random.pt"
            )
        )
    )
)
data = dict(samples_per_gpu=2, workers_per_gpu=2)
#work_dir = "exps/vg/ggnnr50_attr"
work_dir = "exps/vg/ggnnr50_random_attr"
gpu_ids = range(0, 1)
load_from = "exps/vg/r50/latest.pth"
resume_from = None
