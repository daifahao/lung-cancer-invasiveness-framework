import matplotlib.pyplot as plt
import torch
import numpy as np
import SimpleITK as sitk
import pandas as pd
# import Dataset_aug
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
# from torchsampler import ImbalancedDatasetSampler
import torchvision.transforms as transforms

'''Window level/width adjustment'''
def apply_window(image_array, window_center, window_width):
    """
    Apply window level/width adjustment to a CT image.
    Args:
        image_array: np.ndarray, raw CT image (HU values)
        window_center: int, window level (WL)
        window_width: int, window width (WW)
    Returns:
        Windowed image
    """
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    windowed = np.clip(image_array, lower, upper)
    # windowed = (windowed - lower) / (upper - lower)
    return windowed

'''Load a CT/PET volume and convert it to a tensor'''
def load_image(data_path, flag, label=None):
    image = sitk.ReadImage(data_path)
    if flag == 1:
        image = sitk.GetArrayFromImage(image)
        # image = image.astype(np.float32)

        # Clip CT values to a fixed range
        # low = -250
        # high = 400
        # image = np.clip(image, low, high)

        wc, ww = -600.0, 1500.0
        image = apply_window(image, wc, ww)

        # Compute mean and standard deviation
        mean = np.mean(image)
        std = np.std(image)
        # # Normalization
        image = (image - mean) / (std + 1e-8)
        image = image * 5
        # # image = (image - image.min()) / (image.max() - image.min())
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0) # add a channel dimension
        # # Resize with F.interpolate
        # image = F.interpolate(image, size=(224, 224, 224), mode='trilinear', align_corners=False).squeeze(0)

    else:
        image = sitk.GetArrayFromImage(image)

        mean = np.mean(image)
        std = np.std(image)
        # # Print standard deviation for checking
        # # print("Standard deviation:", std)
        image = (image - mean) / (std + 1e-8)
        image = image * 5
        image = torch.tensor(image, dtype=torch.float32).unsqueeze(0)  # add a channel dimension
        # # Resize with F.interpolate
        # image = F.interpolate(image, size=(224, 224, 224), mode='trilinear', align_corners=False).squeeze(0)
    return image

'''Custom dataset class'''
class ReadDataFromCSV(Dataset):
    def __init__(self, csv_path, mode, transform=None, Data_Aug=None, fusion='early'):
        # Transforms
        # self.data = pd.read_csv(csv_path)
        self.data = pd.read_excel(csv_path)
        # self.data = csv_path
        self.transform = transform
        self.Data_Aug = Data_Aug
        self.mode = mode
        self.fusion = fusion

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data.iloc[index]
        label_T = int(sample['p_label'])
        label_N = int(sample['p_label'])  # the CSV/Excel file is expected to have a 'p_label' column storing labels
        label_p = int(sample['p_label'])
        # get image
        if self.mode == 'CT':
            name = str(sample['name'])
            CT_path = sample['CT_Path']  # the table has a 'CT_Path' column storing image paths
            CT_data = load_image(CT_path, flag=1)
            if self.transform:
                # transform_choice = torch.nn.Sequential(*self.transform)[torch.randint(len(self.transform), (1,)).item()]
                transform_choice = self.transform.transforms[torch.randint(len(self.transform.transforms), (1,)).item()]

                CT_data = transform_choice(CT_data)
            if self.Data_Aug:
                CT_data = self.data_aug(CT_data)
                # label = [label] * len(CT_data)  # duplicate the label for each augmented image
            return CT_data, label_T, label_N, name

        elif self.mode == 'PET':
            name = str(sample['name'])
            PET_path = sample['PET_Path']
            PET_data = load_image(PET_path, flag=0)
            if self.transform:
                # transform_choice = torch.nn.Sequential(*self.transform)[torch.randint(len(self.transform), (1,)).item()]
                transform_choice = self.transform.transforms[torch.randint(len(self.transform.transforms), (1,)).item()]

                PET_data = transform_choice(PET_data)
            if self.Data_Aug:
                # CT_data = self.data_aug(CT_data)
                PET_data = self.data_aug(PET_data)
                # label = [label] * len(CT_data)  # duplicate the label for each augmented image
            return  PET_data, label_T, label_N, name

        elif self.mode == 'CP':
            name = str(sample['name'])
            CT_path = sample['CT_Path']  # the table has a 'CT_Path' column storing image paths
            PET_path = sample['PET_Path']
            CT_data = load_image(CT_path, flag=1, label=None)
            PET_data = load_image(PET_path, flag=0, label=None)

            if self.fusion == 'early':
                image = torch.cat((CT_data, PET_data), dim=0)
                if self.transform:
                    # transform_choice = torch.nn.Sequential(*self.transform)[torch.randint(len(self.transform), (1,)).item()]
                    # image = transform_choice(image)
                    image = self.transform(image)
                return image, label_T, label_N, name

            else:
                image = torch.cat((CT_data, PET_data), dim=0)
                if self.transform:
                    # transform_choice = torch.nn.Sequential(*self.transform)[torch.randint(len(self.transform), (1,)).item()]
                    # image = transform_choice(image)
                    image = self.transform(image)
                if self.Data_Aug:
                    CT_data = self.data_aug(CT_data)
                    PET_data = self.data_aug(PET_data)
                    # label = [label] * len(CT_data)  # duplicate the label for each augmented image
                CT_Data = image[0, :, :, :].unsqueeze(0)
                PET_Data = image[1, :, :, :].unsqueeze(0)

                # print("====== __getitem__ DEBUG ======")
                # print("index:", index)
                # print("name:", name, type(name))
                # print("CT_path:", CT_path, type(CT_path))
                # print("PET_path:", PET_path, type(PET_path))
                # print("label_T raw:", label_T, type(label_T))
                # print("label_N raw:", label_N, type(label_N))

                return CT_Data, PET_Data, label_T, label_N, name


    def data_aug(self, ct_slices):
        """
        Augment a series of CT slices with horizontal, vertical and front-back flips.
        ct_slices: a 3D tensor of shape [depth, height, width] representing the CT slice sequence.
        """
        # Define transforms
        flip_horizontally_slices = torch.flip(ct_slices, [3])  # horizontal flip along the width (last dim)
        flip_vertically_slices = torch.flip(ct_slices, [2])  # vertical flip along the height (middle dim)
        flip_both_hv_slices = torch.flip(flip_horizontally_slices, [2])  # horizontal then vertical flip

        # Front-back flip (along the depth axis)
        flip_front_back_slices = torch.flip(ct_slices, [1])  # front-back flip along depth (first dim)

        # Combine front-back flip with 2D flips
        flip_fb_h_slices = torch.flip(flip_front_back_slices, [3])  # front-back + horizontal
        flip_fb_v_slices = torch.flip(flip_front_back_slices, [2])  # front-back + vertical
        flip_fb_hv_slices = torch.flip(flip_fb_h_slices, [2])  # front-back + horizontal + vertical

        return [ct_slices, flip_horizontally_slices, flip_vertically_slices, flip_both_hv_slices,
                flip_front_back_slices, flip_fb_h_slices, flip_fb_v_slices, flip_fb_hv_slices]

if __name__ == '__main__':
    # Example transforms
    # transform = transforms.Compose([
    #     transforms.
    #     transforms.RandomHorizontalFlip(p=1),  # random horizontal flip
    #     transforms.RandomVerticalFlip(p=1),  # random vertical flip
    #     # transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # random translation
    #     # transforms.RandomRotation(90),  # random rotation
    # ])
    # transforms_list = [Dataset_aug.OrininalImage(),
    #                    Dataset_aug.Flip_v(),
    #                    Dataset_aug.Flip_both_hv(),
    #                    Dataset_aug.Flip_both_fh(),
    #                    Dataset_aug.Flip_both_fv(),
    #                    Dataset_aug.Flip_both_hfv(),
    #                    Dataset_aug.Rotate(90)]
    train_path = './Dataset/train.xlsx'
    labels = pd.read_excel(train_path)['T_Label']
    # Count occurrences of each class
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    # Compute weights: each class weight is inversely proportional to its frequency
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in labels]  # assign a weight per sample by label

    # Create WeightedRandomSampler
    weights_tensor = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)

    trainset = ReadDataFromCSV(train_path, transform=None, Data_Aug=0, mode='CP', fusion='late')
    print(trainset)
    train_loader = DataLoader(trainset, 10, shuffle=False, sampler=sampler)

    print(len(train_loader))
    plt.figure(figsize=(12, 20))
    for i, data in enumerate(train_loader):
        # if i <= 2:
        #     continue
        ct, pet, label_T, label_N, PET_path = data
        print(PET_path)
        print(label_T)
        # print(ct.shape)
        print(pet.shape)
        # fig, axes = plt.subplots(5, 2, figsize=(10, 10))
        # Create subplots for CT and PET
        # ct_slice = ct[ct.shape[0] // 2]  # Middle slice of CT
        # pet_slice = pet[pet.shape[0] // 2]  # Middle slice of PET
        for j in range(10):
            plt.subplot(5, 2, j  + 1)  # CT images
            plt.imshow(pet[j, 0, 60, :, :], cmap='gray')
            # plt.title(f'PET Image: {PET_path[j]}')
            plt.axis('off')

            # plt.subplot(5, 2, j * 2 + 2)  # PET images
            # plt.imshow(pet[j, 0, 48, :, :], cmap='gray')
            # # plt.title(f'PET Image: {PET_path[i]}')
            # plt.axis('off')
        plt.savefig('test2.jpg')
        plt.show()
        break
