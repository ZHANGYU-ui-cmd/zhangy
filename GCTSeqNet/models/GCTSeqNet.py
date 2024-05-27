from copy import deepcopy

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import boxes as box_ops

from models.oim import OIMLoss, LOIMLoss
from models.resnet import build_resnet
from utils.utils import *
from models.attention import GCT


class GCTSeqNet(nn.Module):
    def __init__(self, cfg):
        super(GCTSeqNet, self).__init__()

        backbone, box_head = build_resnet(name="resnet50", pretrained=True)
        senet = GCT(num_channels=1024)
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256, 512),), aspect_ratios=((0.5, 1.0, 2.0),)
        )
        head = RPNHead(
            in_channels=backbone.out_channels,
            num_anchors=anchor_generator.num_anchors_per_location()[0],
        )
        pre_nms_top_n = dict(
            training=cfg.MODEL.RPN.PRE_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.PRE_NMS_TOPN_TEST
        )
        post_nms_top_n = dict(
            training=cfg.MODEL.RPN.POST_NMS_TOPN_TRAIN, testing=cfg.MODEL.RPN.POST_NMS_TOPN_TEST
        )
        rpn = RegionProposalNetwork(
            anchor_generator=anchor_generator,
            head=head,
            fg_iou_thresh=cfg.MODEL.RPN.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.RPN.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.RPN.POS_FRAC_TRAIN,
            pre_nms_top_n=pre_nms_top_n,
            post_nms_top_n=post_nms_top_n,
            nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
        )

        faster_rcnn_predictor = FastRCNNPredictor(2048, 2)
        reid_head = deepcopy(box_head)
        box_roi_pool = MultiScaleRoIAlign(
            featmap_names=["feat_res4"], output_size=14, sampling_ratio=2
        )
        box_predictor = BBoxRegressor(2048, num_classes=2, bn_neck=cfg.MODEL.ROI_HEAD.BN_NECK)
        roi_heads = SeqRoIHeads(
            # OIM
            num_pids=cfg.MODEL.LOSS.LUT_SIZE,
            num_cq_size=cfg.MODEL.LOSS.CQ_SIZE,
            oim_momentum=cfg.MODEL.LOSS.OIM_MOMENTUM,
            oim_scalar=cfg.MODEL.LOSS.OIM_SCALAR,
            oim_type=cfg.MODEL.LOSS.TYPE,
            iou_type=cfg.MODEL.LOSS.IOU_TYPE,
            oim_eps=cfg.MODEL.LOSS.OIM_EPS,
            # GCTSeqNet
            faster_rcnn_predictor=faster_rcnn_predictor,
            reid_head=reid_head,
            # parent class
            box_roi_pool=box_roi_pool,
            box_head=box_head,
            box_predictor=box_predictor,
            fg_iou_thresh=cfg.MODEL.ROI_HEAD.POS_THRESH_TRAIN,
            bg_iou_thresh=cfg.MODEL.ROI_HEAD.NEG_THRESH_TRAIN,
            batch_size_per_image=cfg.MODEL.ROI_HEAD.BATCH_SIZE_TRAIN,
            positive_fraction=cfg.MODEL.ROI_HEAD.POS_FRAC_TRAIN,
            bbox_reg_weights=None,
            score_thresh=cfg.MODEL.ROI_HEAD.SCORE_THRESH_TEST,
            nms_thresh=cfg.MODEL.ROI_HEAD.NMS_THRESH_TEST,
            detections_per_img=cfg.MODEL.ROI_HEAD.DETECTIONS_PER_IMAGE_TEST,
        )

        transform = GeneralizedRCNNTransform(
            min_size=cfg.INPUT.MIN_SIZE,
            max_size=cfg.INPUT.MAX_SIZE,
            image_mean=[0.485, 0.456, 0.406],
            image_std=[0.229, 0.224, 0.225],
        )

        self.backbone = backbone
        self.senet = senet
        self.rpn = rpn
        self.roi_heads = roi_heads
        self.transform = transform
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))

        # loss weights
        self.lw_rpn_reg = cfg.SOLVER.LW_RPN_REG
        self.lw_rpn_cls = cfg.SOLVER.LW_RPN_CLS
        self.lw_proposal_reg = cfg.SOLVER.LW_PROPOSAL_REG
        self.lw_proposal_cls = cfg.SOLVER.LW_PROPOSAL_CLS
        self.lw_box_reg = cfg.SOLVER.LW_BOX_REG
        self.lw_box_cls = cfg.SOLVER.LW_BOX_CLS
        self.lw_box_reid = cfg.SOLVER.LW_BOX_REID

        self.local_senet = GCT(num_channels=1024)
        self.local_cc = nn.Conv2d(1024, 256, 1)
        self.local_fc_1 = nn.Linear(256, 32, bias=True)
        self.local_fc_2 = nn.Linear(256, 64, bias=True)
        self.local_fc_3 = nn.Linear(256, 32, bias=True)


    def inference(self, images, targets=None, query_img_as_gallery=False):
        """
        query_img_as_gallery: Set to True to detect all people in the query image.
            Meanwhile, the gt box should be the first of the detected boxes.
            This option serves CBGM.
        """
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)

        features_atten = self.senet(features['feat_res4'])   ###现在应该是[1, 1024, 58, 76]
        features_atten = self.avg_pool(features_atten)    ###变为[1,1024,1,1]

        if query_img_as_gallery:
            assert targets is not None

        if targets is not None and not query_img_as_gallery:
            # query
            boxes = [t["boxes"] for t in targets]
            box_features = self.roi_heads.box_roi_pool(features, boxes, images.image_sizes)
            box_features = self.roi_heads.reid_head(box_features)
            batch_pro_nums = [b.size(0) for b in boxes]
            #### local feature
            split_boxes = box_split(boxes)  ####每一张图片都是[128,3,4]
            boxes_three = [[], [], []]
            for sp in range(3):  ###遍历上、中、下
                for b in range(len(split_boxes)):  ####遍历每张图片
                    ####boxes_three 为list为3的，分别为上、中、下，每一个为（128，4）
                    boxes_three[sp].append(
                        split_boxes[b].transpose(0, 1)[sp])  ###boxes_three将所有图像的上、中、下进行存储   boxes_three list为3
            ##boxes_three:是一个list为3的，该list分别是上、中、下，其中上是一个list为图片张数，每个里面为（128，4）    （3(B，128，4)）
            ###box_split_feats
            box_split_feats = [self.roi_heads.box_roi_pool(features, s_box, images.image_sizes) for s_box in
                               boxes_three]  ###（list3(256，1024，14，14)）
            box_split_feats = torch.stack(box_split_feats).transpose(0, 1)  ####(256,3,1024,14,14)
            B, N, H, W = box_split_feats.size(0), box_split_feats.size(2), box_split_feats.size(
                3), box_split_feats.size(4)
            box_split_feats = box_split_feats.reshape(B * 3, N, H, W)  # (768,1024,14,14)
            box_split_feats = self.local_senet(box_split_feats)  # (768,1024,14,14)
            box_split_feats = F.adaptive_avg_pool2d(box_split_feats, (1, 1))  # (768,1024,1,1)
            feat_medi = torch.relu((self.local_cc(box_split_feats.reshape(B*3, -1, 1, 1)))).reshape(B, 3, -1)
            local_feature = torch.cat((self.local_fc_1(feat_medi[:, 0, :]), self.local_fc_2(feat_medi[:, 1, :]), self.local_fc_3(feat_medi[:, 2, :])), dim=-1)


            embeddings, _ = self.roi_heads.embedding_head(box_features, features_atten, batch_pro_nums, local_feature)
            return embeddings.split(1, 0)
        else:
            # gallery
            proposals, _ = self.rpn(images, features, targets)
            detections, _ = self.roi_heads(
                features, proposals, images.image_sizes, features_atten, targets, query_img_as_gallery
            )
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            return detections

    def forward(self, images, targets=None, query_img_as_gallery=False):
        if not self.training:
            return self.inference(images, targets, query_img_as_gallery)

        images, targets = self.transform(images, targets)   ##(2,3,1216,1504)
        features = self.backbone(images.tensors)   #'feat_res4:(2,1024,76,94)'

        features_atten = self.senet(features['feat_res4'])   ###现在应该是[1, 1024, 58, 76]   ###(2,1024,76,94)
        features_atten = self.avg_pool(features_atten)    ###变为[1,1024,1,1]   ###(2,1024,1,1)

        proposals, proposal_losses = self.rpn(images, features, targets)

        _, detector_losses = self.roi_heads(features, proposals, images.image_sizes, features_atten, targets)

        # rename rpn losses to be consistent with detection losses
        proposal_losses["loss_rpn_reg"] = proposal_losses.pop("loss_rpn_box_reg")
        proposal_losses["loss_rpn_cls"] = proposal_losses.pop("loss_objectness")

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        # apply loss weights
        losses["loss_rpn_reg"] *= self.lw_rpn_reg
        losses["loss_rpn_cls"] *= self.lw_rpn_cls
        losses["loss_proposal_reg"] *= self.lw_proposal_reg
        losses["loss_proposal_cls"] *= self.lw_proposal_cls
        losses["loss_box_reg"] *= self.lw_box_reg
        losses["loss_box_cls"] *= self.lw_box_cls
        losses["loss_box_reid"] *= self.lw_box_reid
        return losses


class SeqRoIHeads(RoIHeads):
    def __init__(
        self,
        num_pids,
        num_cq_size,
        oim_momentum,
        oim_scalar,
        oim_type,
        iou_type,
        oim_eps,
        faster_rcnn_predictor,
        reid_head,
        *args,
        **kwargs
    ):
        super(SeqRoIHeads, self).__init__(*args, **kwargs)
        self.embedding_head = NormAwareEmbedding()
        self.faster_rcnn_predictor = faster_rcnn_predictor
        self.reid_head = reid_head
        # rename the method inherited from parent class
        self.postprocess_proposals = self.postprocess_detections

        self.local_senet = GCT(num_channels=1024)
        self.local_cc = nn.Conv2d(1024, 256, 1)
        # self.local_fc = nn.Linear(1024, 128, bias=False)
        self.local_fc_1 = nn.Linear(256, 32, bias=True)
        self.local_fc_2 = nn.Linear(256, 64, bias=True)
        self.local_fc_3 = nn.Linear(256, 32, bias=True)
        self.down_conv = torchvision.models.resnet.__dict__["resnet50"](pretrained=True).layer4
        if oim_type == 'OIM':
            self.reid_loss = OIMLoss(512, num_pids, num_cq_size, oim_momentum, oim_scalar)
            # self.reid_loss = OIMLoss(512, num_pids, num_cq_size, oim_momentum, oim_scalar, oim_eps)
        if oim_type == 'LOIM':
            self.reid_loss = LOIMLoss(512, num_pids, num_cq_size, oim_momentum, oim_scalar, oim_eps)

        self.oim_type = oim_type
        self.iou_type = iou_type

    def forward(self, features, proposals, image_shapes, features_atten, targets=None, query_img_as_gallery=False, ):
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if self.training:
            proposals, _, proposal_pid_labels, proposal_reg_targets = self.select_training_samples(
                proposals, targets
            )     ###proposals是list 2个list （2000，4）

        # ------------------- Faster R-CNN head ------------------ #
        proposal_features = self.box_roi_pool(features, proposals, image_shapes)
        proposal_features = self.box_head(proposal_features)
        proposal_cls_scores, proposal_regs = self.faster_rcnn_predictor(
            proposal_features["feat_res5"]
        )

        if self.training:
            boxes = self.get_boxes(proposal_regs, proposals, image_shapes)
            boxes = [boxes_per_image.detach() for boxes_per_image in boxes]
            boxes, _, box_pid_labels, box_reg_targets = self.select_training_samples(boxes, targets)
        else:
            # invoke the postprocess method inherited from parent class to process proposals
            boxes, scores, _ = self.postprocess_proposals(
                proposal_cls_scores, proposal_regs, proposals, image_shapes
            )

        cws = True
        gt_det = None
        if not self.training and query_img_as_gallery:
            # When regarding the query image as gallery, GT boxes may be excluded
            # from detected boxes. To avoid this, we compulsorily include GT in the
            # detection results. Additionally, CWS should be disabled as the
            # confidences of these people in query image are 1
            cws = False
            gt_box = [targets[0]["boxes"]]
            gt_box_features = self.box_roi_pool(features, gt_box, image_shapes)
            gt_box_features = self.reid_head(gt_box_features)
            batch_pro_nums = [b.size(0) for b in gt_box]


            #### local feature
            split_boxes = box_split(gt_box)  ####每一张图片都是[128,3,4]
            boxes_three = [[], [], []]
            for sp in range(3):  ###遍历上、中、下
                for b in range(len(split_boxes)):  ####遍历每张图片
                    ####boxes_three 为list为3的，分别为上、中、下，每一个为（128，4）
                    boxes_three[sp].append(
                        split_boxes[b].transpose(0, 1)[sp])  ###boxes_three将所有图像的上、中、下进行存储   boxes_three list为3
            ##boxes_three:是一个list为3的，该list分别是上、中、下，其中上是一个list为图片张数，每个里面为（128，4）    （3(B，128，4)）
            ###box_split_feats
            box_split_feats = [self.box_roi_pool(features, s_box, image_shapes) for s_box in
                               boxes_three]  ###（list3(256，1024，14，14)）
            box_split_feats = torch.stack(box_split_feats).transpose(0, 1)  ####(256,3,1024,14,14)
            B, N, H, W = box_split_feats.size(0), box_split_feats.size(2), box_split_feats.size(
                3), box_split_feats.size(4)
            box_split_feats = box_split_feats.reshape(B * 3, N, H, W)  # (768,1024,14,14)
            box_split_feats = self.local_senet(box_split_feats)  # (768,1024,14,14)
            box_split_feats = F.adaptive_avg_pool2d(box_split_feats, (1, 1))  # (768,1024,1,1)
            feat_medi = torch.relu((self.local_cc(box_split_feats.reshape(B*3, -1, 1, 1)))).reshape(B, 3, -1)
            local_feature = torch.cat((self.local_fc_1(feat_medi[:, 0, :]), self.local_fc_2(feat_medi[:, 1, :]), self.local_fc_3(feat_medi[:, 2, :])), dim=-1)


            embeddings, _ = self.embedding_head(gt_box_features, features_atten, batch_pro_nums, local_feature)
            gt_det = {"boxes": targets[0]["boxes"], "embeddings": embeddings}

        # no detection predicted by Faster R-CNN head in test phase
        if boxes[0].shape[0] == 0:
            assert not self.training
            boxes = gt_det["boxes"] if gt_det else torch.zeros(0, 4)
            labels = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            scores = torch.ones(1).type_as(boxes) if gt_det else torch.zeros(0)
            embeddings = gt_det["embeddings"] if gt_det else torch.zeros(0, 512)
            return [dict(boxes=boxes, labels=labels, scores=scores, embeddings=embeddings)], []

        # --------------------- Baseline head -------------------- #
        #### local feature
        split_boxes = box_split(boxes)   ####每一张图片都是[128,3,4]
        boxes_three = [[], [], []]
        for sp in range(3):    ###遍历上、中、下
            for b in range(len(split_boxes)):  ####遍历每张图片
                ####boxes_three 为list为3的，分别为上、中、下，每一个为（128，4）
                boxes_three[sp].append(split_boxes[b].transpose(0, 1)[sp])    ###boxes_three将所有图像的上、中、下进行存储   boxes_three list为3
        ##boxes_three:是一个list为3的，该list分别是上、中、下，其中上是一个list为图片张数，每个里面为（128，4）    （3(B，128，4)）
        ###box_split_feats
        box_split_feats = [self.box_roi_pool(features, s_box, image_shapes) for s_box in boxes_three]  ###（list3(256，1024，14，14)）
        box_split_feats = torch.stack(box_split_feats).transpose(0, 1)  ####(256,3,1024,14,14)
        B, N, H, W = box_split_feats.size(0), box_split_feats.size(2), box_split_feats.size(3), box_split_feats.size(4)
        box_split_feats = box_split_feats.reshape(B*3, N, H, W)  #(768,1024,14,14)
        box_split_feats = self.local_senet(box_split_feats)   #(768,1024,14,14)
        box_split_feats = F.adaptive_avg_pool2d(box_split_feats, (1, 1))                        #(768,1024,1,1)


        feat_medi = torch.relu((self.local_cc(box_split_feats.reshape(B*3, -1, 1, 1)))).reshape(B, 3, -1)
        local_feature = torch.cat((self.local_fc_1(feat_medi[:, 0, :]), self.local_fc_2(feat_medi[:, 1, :]), self.local_fc_3(feat_medi[:, 2, :])), dim=-1)

        #### origin seq_net
        ####2个的(256,1024,14,14)
        box_features = self.box_roi_pool(features, boxes, image_shapes)
        box_features = self.reid_head(box_features)       #'feat_res4': (128,1024,1,1)  'feat_res5':(128,2048,1,1)
        box_regs = self.box_predictor(box_features["feat_res5"])    ##(128,8)   ##
        batch_pro_nums = [b.size(0) for b in boxes]
        box_embeddings, box_cls_scores = self.embedding_head(box_features, features_atten, batch_pro_nums, local_feature)  #*
        if box_cls_scores.dim() == 0:
            box_cls_scores = box_cls_scores.unsqueeze(0)

        result, losses = [], {}
        if self.training:
            proposal_labels = [y.clamp(0, 1) for y in proposal_pid_labels]
            box_labels = [y.clamp(0, 1) for y in box_pid_labels]
            losses = detection_losses(
                proposal_cls_scores,
                proposal_regs,
                proposal_labels,
                proposal_reg_targets,
                box_cls_scores,
                box_regs,
                box_labels,
                box_reg_targets,
            )
            if self.oim_type == 'OIM':
                loss_box_reid = self.reid_loss(box_embeddings, box_pid_labels)
            # 1.计算所有box和gt的iou
            # 2.对每个box选择对应最大的gt的iou
            # 3.过滤背景box??
            if self.oim_type == 'LOIM':
                max_iou_list = []
                for batch_index in range(len(boxes)):
                    box_p = boxes[batch_index]
                    box_t = targets[batch_index]['boxes']
                    if self.iou_type == 'IOU':  ##IOU
                        ious = box_ops.box_iou(box_p, box_t)
                    if self.iou_type == 'GIOU':
                        ious = box_ops.generalized_box_iou(box_p, box_t)
                    if self.iou_type == 'DIOU':
                        ious = box_ops.box_diou(box_p, box_t)
                    if self.iou_type == 'CIOU':
                        ious = box_ops.box_ciou(box_p, box_t)
                    ious_max = torch.max(ious, dim=1)[0]
                    max_iou_list.append(ious_max)
                ious = torch.cat(max_iou_list, dim=0)

                ious = torch.clamp(ious, min=0.7)  ##我也不知道为什么要限定在0.7，说是为了安全？？？
                loss_box_reid = self.reid_loss(box_embeddings, box_pid_labels, ious)

            losses.update(loss_box_reid=loss_box_reid)
        else:
            # The IoUs of these boxes are higher than that of proposals,
            # so a higher NMS threshold is needed
            orig_thresh = self.nms_thresh
            self.nms_thresh = 0.5
            boxes, scores, embeddings, labels = self.postprocess_boxes(
                box_cls_scores,
                box_regs,
                box_embeddings,
                boxes,
                image_shapes,
                fcs=scores,
                gt_det=gt_det,
                cws=cws,
            )
            # set to original thresh after finishing postprocess
            self.nms_thresh = orig_thresh
            num_images = len(boxes)
            for i in range(num_images):
                result.append(
                    dict(
                        boxes=boxes[i], labels=labels[i], scores=scores[i], embeddings=embeddings[i]
                    )
                )
        return result, losses

    def get_boxes(self, box_regression, proposals, image_shapes):
        """
        Get boxes from proposals.
        """
        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)
        pred_boxes = pred_boxes.split(boxes_per_image, 0)

        all_boxes = []
        for boxes, image_shape in zip(pred_boxes, image_shapes):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)
            # remove predictions with the background label
            boxes = boxes[:, 1:].reshape(-1, 4)
            all_boxes.append(boxes)

        return all_boxes

    def postprocess_boxes(
        self,
        class_logits,
        box_regression,
        embeddings,
        proposals,
        image_shapes,
        fcs=None,
        gt_det=None,
        cws=True,
    ):
        """
        Similar to RoIHeads.postprocess_detections, but can handle embeddings and implement
        First Classification Score (FCS).
        """
        device = class_logits.device

        boxes_per_image = [len(boxes_in_image) for boxes_in_image in proposals]
        pred_boxes = self.box_coder.decode(box_regression, proposals)

        if fcs is not None:
            # Fist Classification Score (FCS)
            pred_scores = fcs[0]
        else:
            pred_scores = torch.sigmoid(class_logits)
        if cws:
            # Confidence Weighted Similarity (CWS)
            embeddings = embeddings * pred_scores.view(-1, 1)

        # split boxes and scores per image
        pred_boxes = pred_boxes.split(boxes_per_image, 0)
        pred_scores = pred_scores.split(boxes_per_image, 0)
        pred_embeddings = embeddings.split(boxes_per_image, 0)

        all_boxes = []
        all_scores = []
        all_labels = []
        all_embeddings = []
        for boxes, scores, embeddings, image_shape in zip(
            pred_boxes, pred_scores, pred_embeddings, image_shapes
        ):
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            # create labels for each prediction
            labels = torch.ones(scores.size(0), device=device)

            # remove predictions with the background label
            boxes = boxes[:, 1:]
            scores = scores.unsqueeze(1)
            labels = labels.unsqueeze(1)

            # batch everything, by making every class prediction be a separate instance
            boxes = boxes.reshape(-1, 4)
            scores = scores.flatten()
            labels = labels.flatten()

            embeddings = embeddings.reshape(-1, 512)

            # remove low scoring boxes
            inds = torch.nonzero(scores > self.score_thresh).squeeze(1)
            boxes, scores, labels, embeddings = (
                boxes[inds],
                scores[inds],
                labels[inds],
                embeddings[inds],
            )

            # remove empty boxes
            keep = box_ops.remove_small_boxes(boxes, min_size=1e-2)
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            if gt_det is not None:
                # include GT into the detection results
                boxes = torch.cat((boxes, gt_det["boxes"]), dim=0)
                labels = torch.cat((labels, torch.tensor([1.0]).to(device)), dim=0)
                scores = torch.cat((scores, torch.tensor([1.0]).to(device)), dim=0)
                embeddings = torch.cat((embeddings, gt_det["embeddings"]), dim=0)

            # non-maximum suppression, independently done per class
            keep = box_ops.batched_nms(boxes, scores, labels, self.nms_thresh)
            # keep only topk scoring predictions
            keep = keep[: self.detections_per_img]
            boxes, scores, labels, embeddings = (
                boxes[keep],
                scores[keep],
                labels[keep],
                embeddings[keep],
            )

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)
            all_embeddings.append(embeddings)

        return all_boxes, all_scores, all_embeddings, all_labels


class NormAwareEmbedding(nn.Module):
    """
    Implements the Norm-Aware Embedding proposed in
    Chen, Di, et al. "Norm-aware embedding for efficient person search." CVPR 2020.
    """

    def __init__(self, featmap_names=["feat_res4", "feat_res5"], in_channels=[1024, 2048], dim=256):  #batch_p
        super(NormAwareEmbedding, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = in_channels
        self.dim = dim

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_channel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            proj = nn.Sequential(nn.Linear(in_channel, indv_dim), nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)

        self.fc = nn.Linear(1024, 128)
       # self.bn = nn.BatchNorm1d(1)

    def forward(self, featmaps, features_atten, batch_pro_nums, local_feature):
        """
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        """
        assert len(featmaps) == len(self.featmap_names)
        if len(featmaps) == 1:
            k, v = featmaps.items()[0]
            v = self._flatten_fc_input(v)

            proposal_num = v.size(0)
            features_atten = features_atten.flatten(start_dim=1)  ####去掉1，1  得到[1,1024]
            features_atten = features_atten.repeat(proposal_num, 1)  ###复制，得到[128, 1024]
            features_atten = self.bn(self.fc(features_atten))  ##将其变为128，128

            embeddings = self.projectors[k](v)
            embeddings = torch.cat((embeddings, features_atten), dim=1)

            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms
        else:

            features_atten = features_atten.flatten(start_dim=1)  ####去掉1，1  得到[1,1024]
#            features_atten = self.bn(self.fc(features_atten))  ##将其变为128，128
            features_atten = self.fc(features_atten)  ##将其变为1，128
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(self.projectors[k](v))   ##此时，outputs为两个（128，128）
            embeddings = torch.cat(outputs, dim=1)
            features_atten_list = [features_atten[m_i].repeat(m, 1) for m_i, m in enumerate(batch_pro_nums)]


            features_atten = torch.cat(features_atten_list, dim=0)


            embeddings = torch.cat((embeddings, features_atten), dim=1)   ###拼接起来是[128,384]
            # print(local_feature.shape[:2])
            if list(local_feature.shape[:2]) == [sum(batch_pro_nums), 128]:
                embeddings = torch.cat((embeddings, local_feature), dim = 1)    ###拼接起来应该是[128,512
            else:
                print("出错咯")
            norms = embeddings.norm(2, 1, keepdim=True)
            embeddings = embeddings / norms.expand_as(embeddings).clamp(min=1e-12)
            norms = self.rescaler(norms).squeeze()
            return embeddings, norms

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x

    def _split_embedding_dim(self):
        parts = len(self.in_channels)
        tmp = [self.dim // parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp



class BBoxRegressor(nn.Module):
    """
    Bounding box regression layer.
    """

    def __init__(self, in_channels, num_classes=2, bn_neck=True):
        """
        Args:
            in_channels (int): Input channels.
            num_classes (int, optional): Defaults to 2 (background and pedestrian).
            bn_neck (bool, optional): Whether to use BN after Linear. Defaults to True.
        """
        super(BBoxRegressor, self).__init__()
        if bn_neck:
            self.bbox_pred = nn.Sequential(
                nn.Linear(in_channels, 4 * num_classes), nn.BatchNorm1d(4 * num_classes)
            )
            init.normal_(self.bbox_pred[0].weight, std=0.01)
            init.normal_(self.bbox_pred[1].weight, std=0.01)
            init.constant_(self.bbox_pred[0].bias, 0)
            init.constant_(self.bbox_pred[1].bias, 0)
        else:
            self.bbox_pred = nn.Linear(in_channels, 4 * num_classes)
            init.normal_(self.bbox_pred.weight, std=0.01)
            init.constant_(self.bbox_pred.bias, 0)

    def forward(self, x):
        if x.ndimension() == 4:
            if list(x.shape[2:]) != [1, 1]:
                x = F.adaptive_avg_pool2d(x, output_size=1)
        x = x.flatten(start_dim=1)
        bbox_deltas = self.bbox_pred(x)
        return bbox_deltas


def detection_losses(
    proposal_cls_scores,
    proposal_regs,
    proposal_labels,
    proposal_reg_targets,
    box_cls_scores,
    box_regs,
    box_labels,
    box_reg_targets,
):
    proposal_labels = torch.cat(proposal_labels, dim=0)
    box_labels = torch.cat(box_labels, dim=0)
    proposal_reg_targets = torch.cat(proposal_reg_targets, dim=0)
    box_reg_targets = torch.cat(box_reg_targets, dim=0)

    loss_proposal_cls = F.cross_entropy(proposal_cls_scores, proposal_labels)
    loss_box_cls = F.binary_cross_entropy_with_logits(box_cls_scores, box_labels.float())

    # get indices that correspond to the regression targets for the
    # corresponding ground truth labels, to be used with advanced indexing
    sampled_pos_inds_subset = torch.nonzero(proposal_labels > 0).squeeze(1)
    labels_pos = proposal_labels[sampled_pos_inds_subset]
    N = proposal_cls_scores.size(0)
    proposal_regs = proposal_regs.reshape(N, -1, 4)

    loss_proposal_reg = F.smooth_l1_loss(
        proposal_regs[sampled_pos_inds_subset, labels_pos],
        proposal_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_proposal_reg = loss_proposal_reg / proposal_labels.numel()

    sampled_pos_inds_subset = torch.nonzero(box_labels > 0).squeeze(1)
    labels_pos = box_labels[sampled_pos_inds_subset]
    N = box_cls_scores.size(0)
    box_regs = box_regs.reshape(N, -1, 4)

    loss_box_reg = F.smooth_l1_loss(
        box_regs[sampled_pos_inds_subset, labels_pos],
        box_reg_targets[sampled_pos_inds_subset],
        reduction="sum",
    )
    loss_box_reg = loss_box_reg / box_labels.numel()

    return dict(
        loss_proposal_cls=loss_proposal_cls,
        loss_proposal_reg=loss_proposal_reg,
        loss_box_cls=loss_box_cls,
        loss_box_reg=loss_box_reg,
    )
