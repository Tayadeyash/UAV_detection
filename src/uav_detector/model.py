"""RetinaNet builder using torchvision's maintained implementation."""
import torch
from torchvision.models.detection import retinanet_resnet50_fpn_v2
from torchvision.models.detection.retinanet import RetinaNetClassificationHead
from torchvision.models import ResNet50_Weights

def build_model(num_classes=2):
    model = retinanet_resnet50_fpn_v2(weights="DEFAULT")
    in_channels = model.head.classification_head.conv[0][0].in_channels
    num_anchors = model.head.classification_head.num_anchors
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels, num_anchors, num_classes,
        norm_layer=torch.nn.BatchNorm2d,
    )
    return model
