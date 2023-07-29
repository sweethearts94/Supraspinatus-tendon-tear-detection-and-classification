from torchvision.models import vgg16
from torchvision.models import resnet18, resnet50


def make_vgg16():
    vgg = vgg16(False, num_classes=2)
    return vgg

def make_resnet18():
    resnet = resnet18(False, num_classes=2)
    return resnet

def make_resnet50():
    resnet = resnet50(False, num_classes=2)
    return resnet