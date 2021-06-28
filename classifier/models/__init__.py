from classifier.models.resnet2d import ResNet50BB

def build_model(cfg):
    if cfg.MODEL.NAME == "resnet50":
        return ResNet50BB(cfg.MODEL)
    return