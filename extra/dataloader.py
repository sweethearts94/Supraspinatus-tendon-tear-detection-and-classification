from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
from random import shuffle
import cv2 as cv


class Loader(DataLoader):
    def __init__(self, other_path: str, useful_path: str, is_train = True):
        _ratio = 0.9
        _other_path = Path(other_path)
        _useful_path = Path(useful_path)
        _other_list = list(_other_path.glob("**/*.jpg"))[:50000]
        _useful_list = list(_useful_path.glob("**/*.jpg"))[:50000]
        _other_list_len = int(len(_other_list) * 0.9)
        _useful_list_len = int(len(_useful_list) * 0.9)
        if is_train:
            self._path_list = _other_list[:_other_list_len] + _useful_list[:_useful_list_len]
        else:
            self._path_list = _other_list[_other_list_len:] + _useful_list[_useful_list_len:]
        shuffle(self._path_list)
        self._image_set = {}
        for item in _other_list:
            self._image_set[item] = 0
        for item in _useful_list:
            self._image_set[item] = 1
        
    def __transformer(self, image):
        transform_pipe = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomPerspective(0.6, 1),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])
        return transform_pipe(image)

    def __len__(self):
        return len(self._path_list)

    def __getitem__(self, index):
        image = cv.imread(str(self._path_list[index]))
        image = self.__transformer(image)
        return image, self._image_set[self._path_list[index]]
    