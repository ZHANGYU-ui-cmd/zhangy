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
from models.RFGCTCasNet import RFGCTCasNet
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

def main(args):
    cfg = get_default_cfg()
    if args.cfg_file:
        cfg.merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    device = torch.device(cfg.DEVICE)
    model = RFGCTCasNet(cfg)
    model.to(device)
    model.eval()

    resume_from_ckpt(args.ckpt, model)


    query_info = osp.join('data/PRW', "query_info.txt")
    with open(query_info, "rb") as f:
        raw = f.readlines()

    vis_path = "PRW_vis"
    os.makedirs(vis_path, exist_ok=True)
    queries_gallerys = defaultdict(list)
    for line in raw:
        linelist = str(line, "utf-8").split(" ")
        pid = int(linelist[0])
        img_name = linelist[5][:-2] + ".jpg"

        x, y, w, h = (
            float(linelist[1]),
            float(linelist[2]),
            float(linelist[3]),
            float(linelist[4]),
        )
        roi = np.array([x, y, x + w, y + h]).astype(np.int32)
        roi = np.clip(roi, 0, None).tolist()  # several coordinates are negative    
        queries_gallerys[pid].append((img_name, roi))
    
    for query, gallerys in tqdm(queries_gallerys.items()):
        query_path = os.path.join(vis_path, str(query))
        os.makedirs(query_path, exist_ok=True)

        query_img_path = os.path.join(query_path, str(query) + '.jpg')
        query_save = crop_image(f"data/PRW/frames/{gallerys[0][0]}", gallerys[0][1])
        cv2.imwrite(query_img_path, query_save)

        query_img = [F.to_tensor(Image.open(f"data/PRW/frames/{gallerys[0][0]}").convert("RGB")).to(device)]
        query_target = [{"boxes":torch.tensor([gallerys[0][1]]).to(device)}]
        query_feat = model(query_img, query_target)[0]
        
        for gal in gallerys:
            gal_path = os.path.join(query_path, str(query) + '_' + gal[0])
            # gallery_img = [F.to_tensor(Image.open(f"data/PRW/query_box/{query}_{gal[0]}").convert("RGB")).to(device)]
            gallery_img = [F.to_tensor(Image.open(f"data/PRW/frames/{gal[0]}").convert("RGB")).to(device)]
            gallery_output = model(gallery_img)[0]
            detections = gallery_output["boxes"]
            gallery_feats = gallery_output["embeddings"]

            similarities = gallery_feats.mm(query_feat.view(-1, 1)).squeeze(1)
            visualize_result(f"data/PRW/frames/{gal[0]}", gal_path, detections.cpu().numpy(), similarities)


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
