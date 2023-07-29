import torch
import torchvision
import json
import cv2 as cv
import numpy as np
import os
import statistic.common as s_common
from net.part1 import EnhancedNet, Bottleneck
from extra.model import make_vgg16, make_resnet18
from config.model import PART1_INPUT_SIZE
from pathlib import Path, PurePath
from torch.utils.data.dataloader import DataLoader
from data.process.part1 import Loader
from tqdm import tqdm

np.set_printoptions(suppress=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

c_threshold = 0.5
main_threshold = {
    "tear_class2": [0.3, 0.3, 0.3, 0.3, 0.3, 0.3],
    "tear_class1": [0.4, 0.4, 0.4, 0.4],
    "rotator_cuff": 0.5,
    "sag_cor": 0.5
}
main_ratio = {
    "rotator_cuff": [0.8, 0.1, 0.1],
    "tear_class1": [0.8, 0.2]
}

CLASSIFICATION_MODLE = ""
MAIN_MODEL = ""
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
LABEL_FILE = ""

def load_classification_model():
    model = make_resnet18()
    checkpoint = torch.load(CLASSIFICATION_MODLE)
    model.load_state_dict(checkpoint)
    return model

def load_main_model():
    model = EnhancedNet(Bottleneck, [3, 4, 3, 2], [2, 3, 2, 4, 6])
    checkpoint = torch.load(MAIN_MODEL)
    model.load_state_dict(checkpoint)
    return model

def preprocess(img: np.ndarray):
    torch_pipeline = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.RandomPerspective(0.6, 1),
        torchvision.transforms.Resize(PART1_INPUT_SIZE),
        torchvision.transforms.ToTensor()
    ])
    return torch_pipeline(img)

def softmax(x, t=1):
    x = x / t
    y = np.exp(x - np.max(x))
    f_x = y / np.sum(np.exp(x))
    return f_x

def eval_pipeline(main_model: torch.nn.Module, c_model: torch.nn.Module, image_tensor) -> dict:
    with torch.no_grad():
        image = image_tensor.to(DEVICE)
        c_model_output = c_model(image)
        main_model_output = main_model(image)
        c_model_output_numpy = c_model_output.cpu().numpy()[0]
        main_model_output_numpy = [ item.cpu().numpy()[0] for item in main_model_output ]
        for index in range(len(main_model_output_numpy)):
            main_model_output_numpy[index] = softmax(main_model_output_numpy[index], 0.1)
        main_model_output_numpy[0] = main_model_output_numpy[0] * main_ratio["rotator_cuff"][0] + np.array([main_model_output_numpy[3][0], np.sum(main_model_output_numpy[3][1:]).item()]) * main_ratio["rotator_cuff"][1] + np.array([np.sum(main_model_output_numpy[4][0:2]).item(), np.sum(main_model_output_numpy[4][2:]).item()]) * main_ratio["rotator_cuff"][2]
        main_model_output_numpy[3] = main_model_output_numpy[3] * main_ratio["tear_class1"][0] + np.array([main_model_output_numpy[4][0], main_model_output_numpy[4][1] / 2, np.sum(main_model_output_numpy[4][2:]).item(), main_model_output_numpy[4][1] / 2]) * main_ratio["tear_class1"][1]
        label = ["ROTATOR CUFF", "right_left", "COR/SAG", "TEAR CLASS1", "TEAR CLASS2"]
        eval_result = {label[index]: np.argmax(main_model_output_numpy[index]) for index in range(len(label))}
        return eval_result

if __name__ == "__main__":
    main_model = load_main_model()
    c_model = load_classification_model()
    main_model.to(DEVICE)
    c_model.to(DEVICE)
    main_model.eval()
    c_model.eval()
    test_loader = Loader(False, 'test')
    test_dt = DataLoader(test_loader, batch_size=1, shuffle=False, num_workers=20)
    process_bar = tqdm(test_dt)
    cms = {
        "ROTATOR CUFF": np.zeros((2,2)),
        "TEAR CLASS1": np.zeros((4, 4)),
        "TEAR CLASS2": np.zeros((6, 6)),
        "COR/SAG": np.zeros((2, 2))
    }
    num_classes = [2, 3, 2, 4, 6]
    key_map = {"ROTATOR CUFF": 0, "COR/SAG": 2, "TEAR CLASS1": 3, "TEAR CLASS2": 4}
    for _, (img_tensor, label_tensor, img_filepath) in enumerate(process_bar):
        eval_result = eval_pipeline(main_model, c_model, img_tensor)
        for key in key_map.keys():
            cms[key] += s_common.confusion_matrix(label_tensor[:, key_map[key]], torch.Tensor(list(eval_result.values()))[key_map[key]], num_classes[key_map[key]])
    for key, value in cms.items():
        print("#####  {}  #####".format(key))
        statistics_func_map = {
            'acc': s_common.acc, 
            'recall': s_common.recall,
            'ppv': s_common.precision,
            'npv': s_common.npv,
            'youden': s_common.youden_index,
        }
        for s_key, s_value in statistics_func_map.items():
            temp_list = []
            for n_index in range(num_classes[key_map[key]]):
                temp_list.append(s_value(value, n_index))
            print("{}: {}".format(s_key, temp_list))
        print("cm: \n{}".format(value))