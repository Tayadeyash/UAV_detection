"""Dataset/model sanity evaluation and detection metrics helper.
Requires: torchmetrics[ detection ] for full mAP metrics.
"""
import argparse, json, torch
from torch.utils.data import DataLoader
from src.uav_detector.dataset import VOCDataset, collate_fn
from src.uav_detector.model import build_model

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='dataset'); p.add_argument('--weights',default='checkpoints/best.pt'); p.add_argument('--score',type=float,default=.35); args=p.parse_args()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=VOCDataset(args.data,train=False); loader=DataLoader(ds,1,shuffle=False,collate_fn=collate_fn)
    model=build_model().to(device); ckpt=torch.load(args.weights,map_location=device,weights_only=False); model.load_state_dict(ckpt['model']); model.eval()
    total_gt=total_pred=matched=0
    with torch.inference_mode():
        for images,targets in loader:
            outputs=model([images[0].to(device)])[0]
            keep=outputs['scores'] >= args.score
            boxes=outputs['boxes'][keep].cpu(); gt=targets[0]['boxes']
            total_gt += len(gt); total_pred += len(boxes)
            if len(boxes) and len(gt):
                iou=torchvision_iou(boxes,gt); matched += int((iou.max(1).values >= .5).sum())
    precision=matched/max(1,total_pred); recall=matched/max(1,total_gt)
    print(json.dumps({'images':len(ds),'ground_truth_boxes':total_gt,'predictions':total_pred,'matched_iou50':matched,'precision_proxy':precision,'recall_proxy':recall},indent=2))

def torchvision_iou(a,b):
    from torchvision.ops import box_iou
    return box_iou(a,b)
if __name__=='__main__': main()
