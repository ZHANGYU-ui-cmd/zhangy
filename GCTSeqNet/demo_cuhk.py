from scipy.io import loadmat
import argparse
from glob import glob
import os.path as osp
import os
import cv2
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import torch
import torch.utils.data
from PIL import Image
from torchvision.transforms import functional as F
from tqdm import tqdm

from defaults import get_default_cfg
from models.GCTSeqNet import GCTSeqNet
from utils.utils import resume_from_ckpt


def visualize_result(img_path, save_path, detections, similarities):
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.imshow(plt.imread(img_path))
    plt.axis("off")
    for detection, sim in zip(detections, similarities):
        x1, y1, x2, y2 = detection
        ax.add_patch(
            plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#4CAF50", linewidth=3.5
            )
        )
        ax.add_patch(
            plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="white", linewidth=1)
        )
        ax.text(
            x1 + 5,
            y1 - 18,
            "{:.2f}".format(sim),
            bbox=dict(facecolor="#4CAF50", linewidth=0),
            fontsize=20,
            color="white",
        )
    # plt.tight_layout()
    fig.savefig(save_path)
    # plt.show()
    plt.close(fig)

def crop_image(image_path, roi_box):
    """
    根据ROI区域框进行图像裁剪
    :param image_path: 图像文件路径
    :param roi_box: ROI区域的坐标，格式为[x1, y1, x2, y2]
    :return: 裁剪后的图像
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("图像文件无法读取或路径不正确")
    
    # 获取ROI区域
    x1, y1, x2, y2 = roi_box
    roi = image[y1:y2, x1:x2]
    
    return roi

def load_split_img_names():
    """
    Load the image names for the specific split.
    """
    # gallery images
    gallery_imgs = loadmat(osp.join("data/CUHK-SYSU/annotation", "pool.mat"))
    gallery_imgs = gallery_imgs["pool"].squeeze()
    gallery_imgs = [str(a[0]) for a in gallery_imgs]
    
    return gallery_imgs

def load_annotations():
    # load all images and build a dict from image to boxes
    all_imgs = loadmat(osp.join("data/CUHK-SYSU/annotation", "Images.mat"))
    all_imgs = all_imgs["Img"].squeeze()
    name_to_boxes = {}
    name_to_pids = {}
    unlabeled_pid = 5555  # default pid for unlabeled people
    for img_name, _, boxes in all_imgs:
        img_name = str(img_name[0])
        boxes = np.asarray([b[0] for b in boxes[0]])
        boxes = boxes.reshape(boxes.shape[0], 4)  # (x1, y1, w, h)
        valid_index = np.where((boxes[:, 2] > 0) & (boxes[:, 3] > 0))[0]
        assert valid_index.size > 0, "Warning: {} has no valid boxes.".format(img_name)
        boxes = boxes[valid_index]
        name_to_boxes[img_name] = boxes.astype(np.int32)
        name_to_pids[img_name] = unlabeled_pid * np.ones(boxes.shape[0], dtype=np.int32)

    def set_box_pid(boxes, box, pids, pid):
        for i in range(boxes.shape[0]):
            if np.all(boxes[i] == box):
                pids[i] = pid
                return


    protoc = loadmat("data/CUHK-SYSU/annotation/test/train_test/TestG50.mat")
    protoc = protoc["TestG50"].squeeze()
    for index, item in enumerate(protoc):
        # query
        im_name = str(item["Query"][0, 0][0][0])
        box = item["Query"][0, 0][1].squeeze().astype(np.int32)
        set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)
        # gallery
        gallery = item["Gallery"].squeeze()
        for im_name, box, _ in gallery:
            im_name = str(im_name[0])
            if box.size == 0:
                break
            box = box.squeeze().astype(np.int32)
            set_box_pid(name_to_boxes[im_name], box, name_to_pids[im_name], index + 1)

    annotations = []
    imgs = load_split_img_names()
    for img_name in imgs:
        boxes = name_to_boxes[img_name]
        boxes[:, 2:] += boxes[:, :2]  # (x1, y1, w, h) -> (x1, y1, x2, y2)
        pids = name_to_pids[img_name]
        annotations.append(
            {
                "img_name": img_name,
                "boxes": boxes,
                "pids": pids,
            }
        )

    queries_gallerys = defaultdict(list)
    for annotation in annotations:
        for i, pid in enumerate(annotation['pids']):
            if pid != 5555:
                queries_gallerys[pid].append((annotation['img_name'], annotation['boxes'][i].tolist()))

    return queries_gallerys

def main(args):    
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)

    print("Creating model")
    model = GCTSeqNet(cfg)
    model.to(device)
    model.eval()

    resume_from_ckpt(args.ckpt, model)

    vis_path = "CUHK-SYSU_vis"
    os.makedirs(vis_path, exist_ok=True)
    queries_gallerys = load_annotations()

    for query, gallerys in tqdm(queries_gallerys.items()):
        query_path = os.path.join(vis_path, str(query))
        os.makedirs(query_path, exist_ok=True)
        
        query_img_path = os.path.join(query_path, str(query) + '.jpg')
        query_save = crop_image(f"data/CUHK-SYSU/Image/SSM/{gallerys[0][0]}", gallerys[0][1])
        cv2.imwrite(query_img_path, query_save)

        query_img = [F.to_tensor(Image.open(f"data/CUHK-SYSU/Image/SSM/{gallerys[0][0]}").convert("RGB")).to(device)]
        query_target = [{"boxes":torch.tensor([gallerys[0][1]]).to(device)}]
        query_feat = model(query_img, query_target)[0]
        
        for gal in gallerys:
            gal_path = os.path.join(query_path, str(query) + '_' + gal[0])
            # gallery_img = [F.to_tensor(Image.open(f"data/PRW/query_box/{query}_{gal[0]}").convert("RGB")).to(device)]
            gallery_img = [F.to_tensor(Image.open(f"data/CUHK-SYSU/Image/SSM/{gal[0]}").convert("RGB")).to(device)]
            gallery_output = model(gallery_img)[0]
            detections = gallery_output["boxes"]
            gallery_feats = gallery_output["embeddings"]

            similarities = gallery_feats.mm(query_feat.view(-1, 1)).squeeze(1)
            visualize_result(f"data/CUHK-SYSU/Image/SSM/{gal[0]}", gal_path, detections.cpu().numpy(), similarities)







    query_img = [F.to_tensor(Image.open("demo_imgs/query.jpg").convert("RGB")).to(device)]
    query_target = [{"boxes": torch.tensor([[0, 0, 466, 943]]).to(device)}]
    query_feat = model(query_img, query_target)[0]

    gallery_img_paths = sorted(glob("demo_imgs/gallery-*.jpg"))
    for gallery_img_path in gallery_img_paths:
        print(f"Processing {gallery_img_path}")
        gallery_img = [F.to_tensor(Image.open(gallery_img_path).convert("RGB")).to(device)]
        gallery_output = model(gallery_img)[0]
        detections = gallery_output["boxes"]
        gallery_feats = gallery_output["embeddings"]

        # Compute pairwise cosine similarities,
        # which equals to inner-products, as features are already L2-normed
        similarities = gallery_feats.mm(query_feat.view(-1, 1)).squeeze()

        print(detections.cpu().numpy().shape)
        print(similarities.shape)
        assert 0
        visualize_result(gallery_img_path, detections.cpu().numpy(), similarities)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a person search network.")
    parser.add_argument("--cfg", dest="cfg_file", help="Path to configuration file.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint to resume or evaluate.")
    parser.add_argument(
        "opts", nargs=argparse.REMAINDER, help="Modify config options using the command-line"
    )
    args = parser.parse_args()
    with torch.no_grad():
        main(args)
