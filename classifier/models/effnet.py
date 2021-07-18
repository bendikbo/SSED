from efficientnet_pytorch import EfficientNet


def effnet(cfg):
    return EfficientNet.from_pretrained(cfg.MODEL.NAME, num_classes=cfg.MODEL.NUM_CLASSES)
