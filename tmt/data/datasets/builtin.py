import os

from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data.datasets.coco import register_coco_instances

from .ovis import _get_ovis_instances_meta
from tmt.data.datasets.ytvis import (
    register_ytvis_instances,
    _get_ytvis_2019_instances_meta,
    _get_ytvis_2021_instances_meta
)
from tmt.data.datasets.uvo import (
    register_uvo_instances,
    _get_uvo_instances_meta
)

# ==== Predefined splits for YTVIS 2019 ===========
_PREDEFINED_SPLITS_YTVIS_2019 = {
    "ytvis_2019_train": ("/mnt/tmt/ytvis_2019/train/JPEGImages",
                         "/mnt/tmt/ytvis_2019/train.json"),
    "ytvis_2019_val": ("/mnt/tmt/ytvis_2019/valid/JPEGImages",
                       "/mnt/tmt/ytvis_2019/valid.json"),
    "ytvis_2019_test": ("/mnt/tmt/ytvis_2019/test/JPEGImages",
                        "/mnt/tmt/ytvis_2019/test.json"),
    # "ytvis_2019_train": ("ytvis_2019/train/JPEGImages",
    #                      "ytvis_2019/train.json"),
    # "ytvis_2019_val": ("ytvis_2019/valid/JPEGImages",
    #                    "ytvis_2019/valid.json"),
    # "ytvis_2019_test": ("ytvis_2019/test/JPEGImages",
    #                     "ytvis_2019/test.json"),
    # "ytvis_2019_val_all_frames": ("ytvis_2019/valid_all_frames/JPEGImages",
    #                     "ytvis_2019/valid_all_frames.json"),
}


# ==== Predefined splits for YTVIS 2021 ===========
_PREDEFINED_SPLITS_YTVIS_2021 = {
    "ytvis_2021_train": ("/mnt/petrelfs/ytvis_2021/train/JPEGImages",
                         "/mnt/petrelfs/ytvis_2021/train/instances.json"),
    "ytvis_2021_val": ("/mnt/petrelfs/ytvis_2021/valid/JPEGImages",
                       "/mnt/petrelfs/ytvis_2021/valid/instances.json"),
    "ytvis_2021_test": ("/mnt/petrelfs/ytvis_2021/test/JPEGImages",
                        "/mnt/petrelfs/ytvis_2021/test/instances.json"),
}

_PREDEFINED_SPLITS_UVO = {
    "uvo_dense_train_ca": ("/mnt/petrelfs/uvo_videos_dense_frames",
                        "/mnt/petrelfs/VideoDenseSet/UVO_video_train_dense_edit.json"),
    "uvo_dense_val_ca": ("/mnt/petrelfs/uvo_videos_dense_frames",
                      "/mnt/petrelfs/VideoDenseSet/UVO_video_val_dense.json"),
    "uvo_dense_train": ("/mnt/tmt/uvo_videos_dense_frames",
                    "/mnt/tmt/UVO_video_train_dense_with_label_edit.json"),
    "uvo_dense_val": ("/mnt/tmt/uvo_videos_dense_frames", 
                    "/mnt/tmt/UVO_video_val_dense_with_label.json"),    
    # "uvo_dense_test": ("uvo_dense/test/uvo_videos_dense_frames",
    #                    "uvo_dense/VideoDenseSet/UVO_video_test_dense.json"),
    # "uvo_sparse_train": ("uvo_dense/uvo_videos_sparse_frames",
    #                     "uvo_dense/VideoSparseSet/UVO_sparse_train_video.json"),
    # "uvo_sparse_val": ("uvo_dense/uvo_videos_sparse_frames",
    #                   "uvo_dense/VideoSparseSet/UVO_sparse_val_video.json"),
    # "uvo_sparse_test": ("uvo_dense/uvo_videos_sparse_frames",
    #                    "uvo_dense/VideoSparseSet/UVO_sparse_test_video.json"),
    "ytvis_2019_train_ca": ("/mnt/tmt/MaskFormer/datasets/ytvis_2019/train/JPEGImages",
                         "/mnt/petrelfs/ytvis_2019/train_ca.json"),
    "ytvis_2019_val_ca": ("/mnt/tmt/MaskFormer/datasets/ytvis_2019/valid/JPEGImages",
                       "/mnt/petrelfs/ytvis_2019/valid_ca.json"),
}

# ====    Predefined splits for OVIS    ===========
# _PREDEFINED_SPLITS_OVIS = {
#     "ovis_train": ("/mnt/petrelfs/OVIS/Images/train",
#                    "/mnt/petrelfs/OVIS/annotations_train.json"),
#     "ovis_val": ("/mnt/petrelfs/OVIS/Images/valid",
#                  "/mnt/petrelfs/OVIS/annotations_valid.json"),
#     "ovis_test": ("/mnt/petrelfs/OVIS/Images/test",
#                   "/mnt/petrelfs/OVIS/annotations_test.json"),
# }
_PREDEFINED_SPLITS_OVIS = {
    "ovis_train_ca": ("/mnt/petrelfs/OVIS/Images/train_ca",
                   "/mnt/petrelfs/OVIS/Images/train_ca/train.json"),
    "ovis_val_ca": ("/mnt/petrelfs/OVIS/Images/valid_ca",
                 "/mnt/petrelfs/OVIS/Images/valid_ca/valid.json"),
    "ovis_train": ("/mnt/tmt/OVIS/Images/train",
                   "/mnt/tmt/OVIS/annotations_train.json"),
    "ovis_val": ("/mnt/tmt/OVIS/Images/valid",
                 "/mnt/tmt/OVIS/annotations_valid.json"),
    "ovis_test": ("ovis/test",
                  "ovis/annotations/test.json"),
}

_PREDEFINED_SPLITS_COCO_VIDEO = {
    "coco2ytvis2019_train": ("/mnt/tmt/coco/train2017", "/mnt/tmt/coco2ytvis2019_train.json"),
    "coco2ytvis2019_val": ("/mnt/tmt/coco/val2017", "/mnt/tmt/coco2ytvis2019_val.json"),
    "coco2ytvis2019_full_train": ("/mnt/tmt/coco/train2017", "/mnt/petrelfs/coco2ytvis2019_full_train.json"),
    "coco2ytvis2019_full_val": ("/mnt/tmt/coco/val2017", "/mnt/petrelfs/coco2ytvis2019_full_val.json"),
    "coco2ytvis2021_train": ("//mnt/tmt/coco/train2017", "/mnt/petrelfs/coco2ytvis2021_train.json"),
    "coco2ytvis2021_val": ("/mnt/cache/share_data/zhangwenwei/data/coco/val2017", "/mnt/petrelfs/coco2ytvis2021_val.json"),
    "coco2ovis_train": ("/mnt/cache/share_data/zhangwenwei/data/coco/train2017", "/mnt/petrelfs/coco2ovis_train.json"),
    "coco2ovis_val": ("/mnt/cache/share_data/zhangwenwei/data/coco/val2017", "/mnt/petrelfs/coco2ovis_val.json"),
}


def register_all_ytvis_2019(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2019.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2019_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_ytvis_2021(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_YTVIS_2021.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ytvis_2021_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_coco_video(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_COCO_VIDEO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_coco_instances(
            key,
            _get_builtin_metadata("coco"),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


def register_all_uvo_dense(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_UVO.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_uvo_instances(
            key,
            _get_uvo_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        ) 

def register_all_ovis(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_OVIS.items():
        # Assume pre-defined datasets live in `./datasets`.
        register_ytvis_instances(
            key,
            _get_ovis_instances_meta(),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )


if __name__.endswith(".builtin"):
    # Assume pre-defined datasets live in `./datasets`.
    _root = os.getenv("DETECTRON2_DATASETS", "datasets")
    register_all_ovis(_root)
    register_all_uvo_dense(_root)
    register_all_ytvis_2019(_root)
    register_all_ytvis_2021(_root)
    register_all_coco_video(_root)
