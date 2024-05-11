import os
import torch

from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader


class MnistDataset(Dataset):

    def __init__(self, root_dir, transform=None, train: str = 'train', val_split: float = 0.2, random_seed: int = 42):
        """
        要有两个列表，列表里是所有图片路径和所有标签
        root_dir: 数据集的根目录。
        transform: 要应用域每张图片的转换。
        train: 数据集划分方式，train为80%，val为20%，test为100%。
        val_split: 训练数据中用于验证集的比例。
        random_seed: 随机种子。
        """
        self.root_dir = root_dir
        self.transform = transform
        images = []
        labels = []

        for label in sorted(os.listdir(root_dir)):
            label_path = os.path.join(root_dir, label)

            if os.path.isdir(label_path):
                for img_file in os.listdir(label_path):
                    img_path = os.path.join(label_path, img_file)
                    if os.path.isfile(img_path):
                        images.append(img_path)
                        labels.append(int(label))

        if train == 'train':
            self.images, _, self.labels, _ = train_test_split(
                images, labels, test_size=val_split, random_state=random_seed
            )
        elif train == 'eval':
            _, self.images, _, self.labels = train_test_split(
                images, labels, test_size=val_split, random_state=random_seed
            )
        elif train == 'test':
            self.images, self.labels = images, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path)
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_dataset = MnistDataset(root_dir='F:/031-study_direction/02-跨系统日志序列异常检测/002-code/mnist/dataset/mnist/mnist_test', transform=transform, train='test')
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    for image, labels in test_loader:
        print(image.shape)
    print('0')
