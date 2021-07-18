from efficientnet_pytorch import EfficientNet


def effnet(cfg):
    return EfficientNet.from_pretrained(cfg.NAME, num_classes=cfg.NUM_CLASSES)
