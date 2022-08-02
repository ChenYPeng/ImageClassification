import os
import random
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


class Pokemon(Dataset):
    def __init__(self, root, size, split):
        super(Pokemon, self).__init__()

        self.root = root
        self.resize = size
        self.split = split
        self.name2label = {}
        self.image_list = []
        self.label_list = []

        # os.listdir每次返回目录下的文件列表顺序会不一致  sorted排序
        for name in sorted(os.listdir(self.root)):
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = len(self.name2label.keys())
        
        for name in self.name2label.keys(): # bulbasaur charmander mewtwo pikachu squirtle
            self.image_list += self.read_file(os.path.join(self.root, name))

        random.shuffle(self.image_list)
        for image in self.image_list:  # E:/Datasets/pokeman/charmander\00000088.jpg
            name = os.path.dirname(image).split('/')[-1] # charmander
            self.label_list.append(self.name2label[name])

        assert len(self.label_list) == len(self.label_list)

        if self.split == 'train':
            self.images = self.image_list[:int(0.6*len(self.image_list))]
            self.labels = self.label_list[:int(0.6*len(self.label_list))]
            
        if self.split == 'val':
            self.images = self.image_list[int(0.6*len(self.image_list)):int(0.8 * len(self.image_list))]
            self.labels = self.label_list[int(0.6*len(self.label_list)):int(0.8 * len(self.label_list))]

        if self.split == 'test':
            self.images = self.image_list[int(0.8 * len(self.image_list)):]
            self.labels = self.label_list[int(0.8 * len(self.label_list)):]

        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # idx的范围 [0~len(images)]
        # img:'dataset/pokemon\\bulbasaur\\00000000.png'
        # label: 0/1/2/3/4
        images = Image.open(self.images[idx]).convert('RGB')  # string path => image data

        labels = self.labels[idx]

        if self.split == 'train':
            return self.transform_tr(images), labels
        elif self.split == 'val':
            return self.transform_val(images), labels
        elif self.split == 'test':
            return self.transform_ts(images), labels

        return images, labels

    def transform_tr(self, images):
        composed_transforms =  transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.RandomRotation(15),
            transforms.CenterCrop(self.resize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        return composed_transforms(images)

    def transform_val(self, images):
        composed_transforms =  transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return composed_transforms(images)

    def transform_ts(self, images):
        composed_transforms =  transforms.Compose([
            transforms.Resize((self.resize, self.resize)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        return composed_transforms(images)

    def read_file(self, path):
        """从文件夹中读取数据"""
        files_list = os.listdir(path)
        file_path_list = [os.path.join(path, file) for file in files_list]
        return file_path_list

    def is_image_file(self, filename):
        return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])



if __name__ == '__main__':
    import matplotlib.pyplot as plt

    data_set = Pokemon(root=r'E:/Datasets/pokeman/', size=256, split='test')
    data_loader = DataLoader(data_set, batch_size=6, shuffle=True)
    targets_list = ['bulbasaur', 'charmander', 'mewtwo', 'pikachu', 'squirtle']

    for batch_idx, (examples_data, examples_targets) in enumerate(data_loader):
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i+1)
            plt.tight_layout() # 自动调整子图参数，使之填充满整个图像区域
            plt.imshow(examples_data[i][0], interpolation='none')
            plt.title(targets_list[examples_targets[i]])
            plt.xticks([])
            plt.yticks([])
        plt.show()
