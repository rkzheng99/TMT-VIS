from typing import Tuple
import math

import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.structures import Boxes, ImageList, Instances, BitMasks
from detectron2.utils.memory import retry_if_cuda_oom
import pdb

from mask2former.modeling.criterion import SetCriterion
from mask2former.modeling.matcher import HungarianMatcher
from .modeling.tmt_criterion import tmtSetCriterion
from .modeling.tmt_matcher import tmtHungarianMatcher
from .modeling.transformer_decoder.tmt import tmt


@META_ARCH_REGISTRY.register()
class tmt(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        test_topk_per_image: int,
        # tmt
        tmt_module: nn.Module,
        tmt_criterion: nn.Module,
        num_frames: int,
        num_classes: int,
        is_multi_cls: bool,
        apply_cls_thres: float,
        freeze_detector: bool,
        test_run_chunk_size: int,
        test_interpolate_chunk_size: int,
        is_coco: bool,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.test_topk_per_image = test_topk_per_image

        # tmt hyper-parameters
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.tmt_module = tmt_module
        self.tmt_criterion = tmt_criterion
        self.is_multi_cls = is_multi_cls
        self.apply_cls_thres = apply_cls_thres

        if freeze_detector:
            for name, p in self.named_parameters():
                if not "tmt_module" in name:
                    p.requires_grad_(False)
        self.test_run_chunk_size = test_run_chunk_size
        self.test_interpolate_chunk_size = test_interpolate_chunk_size

        self.is_coco = is_coco

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT
        tmt_deep_supervision = cfg.MODEL.tmt.DEEP_SUPERVISION

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT
        sim_weight = cfg.MODEL.tmt.SIM_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            tmt_last_layer_num=cfg.MODEL.tmt.LAST_LAYER_NUM,
        )

        # tmt
        num_classes = sem_seg_head.num_classes
        hidden_dim = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        tmt_module = tmt(cfg=cfg, in_channels=hidden_dim, aux_loss=tmt_deep_supervision)

        # building criterion for tmt inference
        tmt_matcher = tmtHungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )
        tmt_weight_dict = {
            "loss_tmt_ce": class_weight, "loss_tmt_mask": mask_weight, "loss_tmt_dice": dice_weight
        }
        if sim_weight > 0.0:
            tmt_weight_dict["loss_tmt_sim"] = sim_weight

        if tmt_deep_supervision:
            tmt_dec_layers = cfg.MODEL.tmt.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(tmt_dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in tmt_weight_dict.items()})
            tmt_weight_dict.update(aux_weight_dict)
        tmt_losses = ["tmt_labels", "tmt_masks"]
        if sim_weight > 0.0:
            tmt_losses.append("fg_sim")

        tmt_criterion = tmtSetCriterion(
            num_classes, 
            matcher=tmt_matcher, 
            weight_dict=tmt_weight_dict,
            eos_coef=cfg.MODEL.tmt.NO_OBJECT_WEIGHT,
            losses=tmt_losses, 
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
            sim_use_clip=cfg.MODEL.tmt.SIM_USE_CLIP,
        )

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            # tmt
            "tmt_module": tmt_module,
            "tmt_criterion": tmt_criterion,
            "num_frames": cfg.INPUT.SAMPLING_FRAME_NUM,
            "num_classes": num_classes,
            "is_multi_cls": cfg.MODEL.tmt.MULTI_CLS_ON,
            "apply_cls_thres": cfg.MODEL.tmt.APPLY_CLS_THRES,
            "freeze_detector": cfg.MODEL.tmt.FREEZE_DETECTOR,
            "test_run_chunk_size": cfg.MODEL.tmt.TEST_RUN_CHUNK_SIZE,
            "test_interpolate_chunk_size": cfg.MODEL.tmt.TEST_INTERPOLATE_CHUNK_SIZE,
            "is_coco": cfg.DATASETS.TEST[0].startswith("coco"),
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        if self.training:
            return self.train_model(batched_inputs)
        else:
            # NOTE consider only B=1 case.
            return self.inference(batched_inputs[0])

    def train_model(self, batched_inputs):
        images = []
        # idx_list = [[0,1,2,3,4],[1,2]]
        # idx_list=[[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
        #idx_list=[[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],[0,1,2,3,4]]
        idx_list=[[0,1],[1,2],[2,3],[3,4]]
        for video in batched_inputs:
            for frame in video["image"]:
                images.append(frame.to(self.device))
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        features = self.backbone(images.tensor)

        BT = len(images)
        T = self.num_frames if self.training else BT 
        B = BT // T
        # pdb.set_trace()
        outputs, frame_queries, mask_features = self.sem_seg_head(features)

        mask_features = self.tmt_module.tmt_mask_features(mask_features)
        mask_features = mask_features.view(B, self.num_frames, *mask_features.shape[-3:])

        # mask classification target
        # frame_targets, clip_targets = self.prepare_targets(batched_inputs, images)
        frame_targets, frame_targets_list, clip_targets = self.prepare_targets(batched_inputs, images)
        # bipartite matching-based loss
        losses_list = []
        for i in range(len(idx_list)):
            losses_1, fg_indices_1 = self.criterion(outputs[i], frame_targets_list[i])
            losses_list.append(losses_1)        
        losses, fg_indices = self.criterion(outputs[len(idx_list)], frame_targets)

        tmt_outputs = self.tmt_module(frame_queries)
        tmt_outputs["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", tmt_outputs["pred_mask_embed"], mask_features)
        for out in tmt_outputs["aux_outputs"]:
            out["pred_masks"] = torch.einsum("lbqc,btchw->lbqthw", out["pred_mask_embed"], mask_features)

        for i in range(len(idx_list)):
            for k in list(losses_list[i].keys()):
                if k in self.criterion.weight_dict:
                    losses[k+'_'+str(i)] = losses_list[i][k] * self.criterion.weight_dict[k]
                    
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses_list[i].pop(k)        
        # for k in list(losses.keys()):
        #     if k in self.criterion.weight_dict:
        #         losses[k] *= self.criterion.weight_dict[k]
        #     else:
        #         # remove this loss if not specified in `weight_dict`
        #         losses.pop(k)
        tmt_loss_dict = self.tmt_criterion(tmt_outputs, clip_targets, frame_targets, fg_indices)
        tmt_weight_dict = self.tmt_criterion.weight_dict

        for k in tmt_loss_dict.keys():
            if k in tmt_weight_dict:
                tmt_loss_dict[k] *= tmt_weight_dict[k]
        losses.update(tmt_loss_dict)
        return losses

    def prepare_targets(self, targets, images):
        # pdb.set_trace()
        h_pad, w_pad = images.tensor.shape[-2:]
        frame_gt_instances = []
        frame_gt_instances_list = [[],[],[],[],[],[],[],[],[],[],[]]
        #idx_list=[[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4]]
        #idx_list=[[0,1],[0,2],[0,3],[0,4],[1,2],[1,3],[1,4],[2,3],[2,4],[3,4],[0,1,2,3,4]]
        idx_list=[[0,1],[1,2],[2,3],[3,4]]
        # idx_list = [[0,1,2,3,4],[1,2]]
        frame_gt_instances_1 = []
        frame_gt_instances_2 = []
        frame_gt_instances_3 = []
        frame_gt_instances_4 = []
        clip_gt_instances = []

        for targets_per_video in targets:
            _num_instance = len(targets_per_video["instances"][0])
            mask_shape = [_num_instance, self.num_frames, h_pad, w_pad]
            gt_masks_per_video = torch.zeros(mask_shape, dtype=torch.bool, device=self.device)

            gt_classes_per_video = targets_per_video["instances"][0].gt_classes.to(self.device)
            gt_ids_per_video = []
            for f_i, targets_per_frame in enumerate(targets_per_video["instances"]):
                targets_per_frame = targets_per_frame.to(self.device)
                h, w = targets_per_frame.image_size

                _update_cls = gt_classes_per_video == -1
                gt_classes_per_video[_update_cls] = targets_per_frame.gt_classes[_update_cls]
                gt_ids_per_video.append(targets_per_frame.gt_ids)
                if isinstance(targets_per_frame.gt_masks, BitMasks):
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks.tensor
                else: #polygon
                    gt_masks_per_video[:, f_i, :h, :w] = targets_per_frame.gt_masks

            gt_ids_per_video = torch.stack(gt_ids_per_video, dim=1)
            gt_ids_per_video[gt_masks_per_video.sum(dim=(2,3)) == 0] = -1
            valid_bool_frame = (gt_ids_per_video != -1)
            valid_bool_clip = valid_bool_frame.any(dim=-1)

            gt_classes_per_video = gt_classes_per_video[valid_bool_clip].long() # N,
            gt_ids_per_video = gt_ids_per_video[valid_bool_clip].long()         # N, num_frames
            gt_masks_per_video = gt_masks_per_video[valid_bool_clip].float()    # N, num_frames, H, W
            valid_bool_frame = valid_bool_frame[valid_bool_clip]

            if len(gt_ids_per_video) > 0:
                min_id = max(gt_ids_per_video[valid_bool_frame].min(), 0)
                gt_ids_per_video[valid_bool_frame] -= min_id

            clip_gt_instances.append(
                {
                    "labels": gt_classes_per_video, "ids": gt_ids_per_video, "masks": gt_masks_per_video,
                    "video_len": targets_per_video["video_len"], "frame_idx": targets_per_video["frame_idx"],
                }
            )

            for f_i in range(self.num_frames):
                _cls = gt_classes_per_video.clone()
                _ids = gt_ids_per_video[:, f_i].clone()
                _mask = gt_masks_per_video[:, f_i].clone()

                valid = _ids != -1
                for idx in range(len(idx_list)):
                    if f_i in idx_list[idx] :
                        frame_gt_instances_list[idx].append({
                            "labels": _cls[valid],
                            "ids": _ids[valid],
                            "masks": _mask[valid],
                        })              
                frame_gt_instances.append({
                    "labels": _cls[valid],
                    "ids": _ids[valid],
                    "masks": _mask[valid],
                })                

        return frame_gt_instances, frame_gt_instances_list, clip_gt_instances

    def inference(self, batched_inputs):
        frame_queries, mask_features = [], []
        num_frames = len(batched_inputs["image"])
        to_store = self.device if num_frames <= 36 else "cpu"

        for i in range(math.ceil(num_frames / self.test_run_chunk_size)):
            images = batched_inputs["image"][i*self.test_run_chunk_size : (i+1)*self.test_run_chunk_size]
            images = [(x.to(self.device) - self.pixel_mean) / self.pixel_std for x in images]
            images = ImageList.from_tensors(images, self.size_divisibility)

            features = self.backbone(images.tensor)
            outputs, _frame_queries, _mask_features = self.sem_seg_head(features)

            _mask_features = self.tmt_module.tmt_mask_features(_mask_features)

            # BT is 1 as runs per frame
            frame_queries.append(_frame_queries[-1])    # T', fQ, C
            mask_features.append(_mask_features.to(to_store))  # T', C, H, W

        interim_size = images.tensor.shape[-2:]
        image_size = images.image_sizes[0]  # image size without padding after data augmentation

        out_height = batched_inputs.get("height", image_size[0])  # raw image size before data augmentation
        out_width = batched_inputs.get("width", image_size[1])

        del outputs, images, batched_inputs

        frame_queries = torch.cat(frame_queries)[None]  # 1, T, fQ, C
        mask_features = torch.cat(mask_features)        # T, C, H, W

        tmt_outputs = self.tmt_module(frame_queries)

        mask_cls = tmt_outputs["pred_logits"][-1, 0]       # cQ, K+1
        mask_embed = tmt_outputs["pred_mask_embed"][-1, 0] # cQ, C

        del tmt_outputs

        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)

        num_topk = self.test_topk_per_image
        scores_per_video, topk_indices = scores.flatten(0, 1).topk(num_topk, sorted=False)
        labels_per_video = labels[topk_indices]

        topk_indices = torch.div(topk_indices, self.sem_seg_head.num_classes, rounding_mode='floor')
        mask_embed = mask_embed[topk_indices]

        masks_per_video = []
        numerator = torch.zeros(len(mask_embed), dtype=torch.float, device=self.device)
        denominator = torch.zeros(len(mask_embed), dtype=torch.float, device=self.device)
        for i in range(math.ceil(len(mask_features) / self.test_interpolate_chunk_size)):
            m_f = mask_features[i*self.test_interpolate_chunk_size : (i+1)*self.test_interpolate_chunk_size].to(self.device)

            mask_pred = torch.einsum("qc,tchw->qthw", mask_embed, m_f)

            # upsample masks
            mask_pred = retry_if_cuda_oom(F.interpolate)(
                mask_pred,
                size=interim_size,
                mode="bilinear",
                align_corners=False,
            ) # cQ, T, H, W

            mask_pred = mask_pred[:, :, : image_size[0], : image_size[1]]

            interim_mask_soft = mask_pred.sigmoid()
            interim_mask_hard = interim_mask_soft > 0.5

            numerator += (interim_mask_soft.flatten(1) * interim_mask_hard.flatten(1)).sum(1)
            denominator += interim_mask_hard.flatten(1).sum(1)

            mask_pred = F.interpolate(
                mask_pred, size=(out_height, out_width), mode="bilinear", align_corners=False
            ) > 0.
            masks_per_video.append(mask_pred.to(to_store))
        masks_per_video = torch.cat(masks_per_video, dim=1)
        scores_per_video *= (numerator / (denominator + 1e-6))

        if self.is_coco:
            result = Instances((out_height, out_width))

            result.pred_masks = masks_per_video[:, 0].float()   # T=1 for COCO
            result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
            # Uncomment the following to get boxes from masks (this is slow)
            # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

            result.scores = scores_per_video
            result.pred_classes = labels_per_video

            processed_results = [{"instances": result}]
        else:
            processed_results = {
                "image_size": (out_height, out_width),
                "pred_scores": scores_per_video.tolist(),
                "pred_labels": labels_per_video.tolist(),
                "pred_masks": masks_per_video.cpu(),
            }

        return processed_results
