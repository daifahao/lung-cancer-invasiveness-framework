import os
import time
import torch
import random
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

# import Dataset_aug
from model.ResNet18 import CP_ResNet18_mutli_attention
from Data_loader import ReadDataFromCSV
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_curve, auc

seed_value = 42  # set random seed
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # disable hash randomization for reproducibility
torch.manual_seed(seed_value)  # set seed for CPU
torch.cuda.manual_seed(seed_value)  # set seed for the current GPU (single GPU)
torch.cuda.manual_seed_all(seed_value)  # set seed for all GPUs (multi-GPU)
torch.backends.cudnn.deterministic = True

# Check whether a GPU device is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device:', device)

def model_train(model, train_loader, optimizer, criterion, fusion):
    model.train()
    train_loss = 0.0  # initialize loss for each epoch
    correct = 0  # number of correct predictions
    total = 0  # total number of samples
    probabilities = []  # predicted class probabilities
    pred = []
    targets = []  # ground-truth labels
    true_negatives = 0
    false_positives = 0
    true_positives = 0
    false_negatives = 0
    for batch_idx, data in enumerate(train_loader):
        if fusion == 'early':
            image, labels_T, labels_N, name = data
            image, labels_T, labels_N = image.to(device), labels_T.to(device), labels_N.to(device)
            optimizer.zero_grad()  # reset gradients
            outputs = model(image)  # forward pass
        else:
            images_CT, images_PET, labels_T, labels_N, name = data  # CT/PET data and labels
            # Make sure CT and PET batch sizes match (depends on your task and data)
            assert len(images_CT) == len(images_PET), "Labels for CT and PET do not match."
            # Move data to GPU
            images_CT, images_PET, labels_T, labels_N = images_CT.to(device), images_PET.to(device), labels_T.to(device), labels_N.to(device)
            labels = labels_T
            # sigmoid
            labels = labels.view(-1, 1)  # reshape the target tensor to (batch, 1)
            labels = labels.float()
            # Forward pass
            optimizer.zero_grad()  # reset gradients
            outputs = model(images_CT, images_PET)  # forward pass

        loss = criterion(outputs, labels)  # compute loss
        # Optimization
        loss.backward()  # backpropagation
        optimizer.step()  # SGD

        # Predictions and targets
        outputs2 = torch.sigmoid(outputs)
        probabilities.extend(outputs2.detach().cpu().numpy())  # move predictions to CPU as NumPy arrays
        predicted = torch.where(outputs2 >= 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device))
        pred.extend(predicted.detach().cpu().numpy())

        # # softmax
        # _, predicted = torch.max(outputs, 1)
        # probability = F.softmax(outputs, dim=1)
        # probabilities.extend(probability[:, 1].detach().cpu().numpy())

        # Ground-truth labels
        targets.extend(labels.cpu().numpy())

        # Loss
        train_loss += loss.item()
        # Accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        # Update true negatives, false positives, true positives and false negatives
        true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
        false_positives += ((predicted == 1) & (labels == 0)).sum().item()
        true_positives += ((predicted == 1) & (labels == 1)).sum().item()
        false_negatives += ((predicted == 0) & (labels == 1)).sum().item()

    # Calculate metrics after each epoch
    accuracy = 100.0 * correct / total

    # Calculate AUC-ROC
    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)
    avg_loss = train_loss / len(train_loader)
    # Specificity and sensitivity
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return avg_loss, accuracy, specificity, sensitivity, roc_auc

def model_test(net, test_loader, criterion, fusion):
    net.eval()
    test_loss = 0.0  # initialize loss for each epoch
    correct = 0  # number of correct predictions
    total = 0  # total number of samples
    probabilities = []  # predicted class probabilities
    targets = []  # ground-truth labels
    pred = []
    true_negatives = 0
    false_positives = 0
    true_positives = 0
    false_negatives = 0
    incorrect_samples = []  # paths or indices of misclassified samples
    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            if fusion == 'early':
                image, labels_T, labels_N, name = data
                image, labels_T, labels_N = image.to(device), labels_T.to(device), labels_N.to(device)
                outputs = model(image)  # forward pass
            else:
                images_CT, images_PET, labels_T, labels_N, name = data  # CT/PET data and labels
                # Make sure CT and PET batch sizes match (depends on your task and data)
                assert len(images_CT) == len(images_PET), "Labels for CT and PET do not match."
                # Move data to GPU
                images_CT, images_PET, labels_T, labels_N = images_CT.to(device), images_PET.to(device), labels_T.to(device), labels_N.to(device)

                labels = labels_T
                # sigmoid
                labels = labels.view(-1, 1)  # reshape the target tensor to (batch, 1)
                labels = labels.float()
                # Forward pass
                outputs = model(images_CT, images_PET)  # forward pass

            loss = criterion(outputs, labels)  # compute loss

            # sigmoid
            outputs2 = torch.sigmoid(outputs)
            probabilities.extend(outputs2.detach().cpu().numpy())  # move predictions to CPU as NumPy arrays
            predicted = torch.where(outputs2 >= 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device))
            pred.extend(predicted.detach().cpu().numpy())

            # softmax
            # _, predicted = torch.max(outputs, 1)
            # probability = F.softmax(outputs, dim=1)
            # probabilities.extend(probability[:, 1].detach().cpu().numpy())

            # Ground-truth labels
            targets.extend(labels.cpu().numpy())

            # Loss
            test_loss += loss.item()
            # Accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update true negatives, false positives, true positives and false negatives
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()
            # Find misclassified samples
            # incorrect_mask = predicted != labels
            # incorrect_indices = torch.nonzero(incorrect_mask).cpu().numpy()

            # # Save paths or indices of misclassified samples
            # for idx in incorrect_indices:
            #     incorrect_CT = CT_path[int(idx)]
            #     incorrect_PET = PET_path[int(idx)]
            #     incorrect_samples.append(
            #         {'CT_path': incorrect_CT, 'PET_path': incorrect_PET, 'predicted': predicted[idx],
            #          'actual': labels[idx]})

    # Calculate metrics after each epoch
    accuracy = 100.0 * correct / total

    # Calculate AUC-ROC
    fpr, tpr, _ = roc_curve(targets, probabilities)
    roc_auc = auc(fpr, tpr)
    avg_loss = test_loss / (len(test_loader))
    # Specificity and sensitivity
    specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    return avg_loss, accuracy, specificity, sensitivity, roc_auc

if __name__ == '__main__':

    # model = CP_ResNet18(1, 1)
    # model = CP_ResNet18_middle(1, 2)
    # model = CP_ResNet10_early(2, 2)
    model = CP_ResNet18_mutli_attention(1, 1)
    model.to(device)
    # Load an existing model
    # checkpoint = torch.load('./result/checkpoint/best_model.ckpt', map_location=torch.device('cuda:0'))
    # model.load_state_dict(checkpoint['net'])
    # epoch = checkpoint['epoch']
    epoches = 400
    batchsize = 16
    lr = 0.0001

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.01)
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    # Loss function
    # criterion = nn.CrossEntropyLoss()
    # pos_weight = torch.tensor(1.5, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss()

    # Data augmentation transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  # random horizontal flip
        transforms.RandomVerticalFlip(),  # random vertical flip
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # random translation
        transforms.RandomRotation(180),  # random rotation
    ])

    # transforms_list = [
    #     transforms.RandomVerticalFlip(p=1),  # flip along y axis
    #     transforms.RandomHorizontalFlip(p=1),  # flip along x axis
    #     Dataset_aug.OrininalImage(),  # original image
    #     Dataset_aug.Flip_both_fh(),  # yz
    #     Dataset_aug.Flip_both_fv(),  # xy
    #     Dataset_aug.Flip_both_hv(),  # xz
    #     Dataset_aug.Flip_both_hfv(),  # xyz
    #     Dataset_aug.Flip_v(),  # z
    #     # Dataset_aug.Rotate(90),
    #     transforms.RandomRotation(15),
    #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))
    # ]

    # Datasets
    train_csv = './Dataset/train/train.xlsx'
    test_csv = './Dataset/test/test.xlsx'

    Labels = pd.read_excel(train_csv)['T_Label']

    # Custom sampler and weighting strategy
    # Count occurrences of each class
    class_counts = np.bincount(Labels)
    total_samples = len(Labels)
    # Compute weights: each class weight is inversely proportional to its frequency
    class_weights = 1. / class_counts
    sample_weights = [class_weights[label] for label in Labels]  # assign a weight per sample by label

    # Create WeightedRandomSampler
    weights_tensor = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(weights_tensor, num_samples=len(weights_tensor), replacement=True)
    fusion_type = 'late'
    train_set = ReadDataFromCSV(train_csv, transform=transform, Data_Aug=0, mode='CP', fusion=fusion_type)
    test_set = ReadDataFromCSV(test_csv, transform=None, Data_Aug=0, mode='CP', fusion=fusion_type)

    train_loader = DataLoader(train_set, batch_size=batchsize, shuffle=False, sampler=sampler)
    test_loader = DataLoader(test_set, batch_size=batchsize, shuffle=False)

    a = 1
    # type = '1mm'
    # Log file path
    log_path = "./result/log/log{}.csv".format(a)
    # Create the log directory if it does not exist
    log_dir = os.path.dirname(log_path)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    # Create an empty log file if it does not exist
    if not os.path.isfile(log_path):
        open(log_path, 'w').close()
    df = pd.DataFrame(columns=['epoch', 'train_loss', 'train_acc', 'train_spec', 'train_sen', 'train_auc',
                               'test_loss', 'test_acc', 'test_spec', 'test_sen', 'test_auc'])  # column names
    df.to_csv(log_path, index=False)  # path can be changed as needed

    # CNN training loop
    best_auc = 0
    min_loss = 0.6
    model_path = './result/checkpoint/{}'.format(a)
    if not os.path.isdir(model_path):
        os.makedirs(model_path)

    for epoch in range(epoches):
        # if epoch <= 100:
        #     continue
        start_time = time.perf_counter()
        train_loss, train_acc, train_spec, train_sen, train_auc = model_train(model, train_loader, optimizer, criterion, fusion=fusion_type)
        test_loss, test_acc, test_spec, test_sen, test_auc = model_test(model, test_loader, criterion, fusion=fusion_type)
        end_time = time.perf_counter()
        print('Epoch {} took: {}s'.format(epoch + 1, end_time - start_time))

        print('[epoch:%.d/%.d], train_loss: %.3f, train_acc: %.3f, train_spec: %.3f, train_sen: %.3f,  train_auc: %.3f' %
              (epoch + 1, epoches, train_loss, train_acc, train_spec, train_sen, train_auc))

        print('[epoch:%.d/%.d], test_loss: %.3f, test_acc: %.3f, test_spec: %.3f, test_sen: %.3f, test_auc: %.3f' %
              (epoch + 1, epoches, test_loss, test_acc, test_spec, test_sen, test_auc))

        data_csv = [epoch+1, train_loss, train_acc, train_spec, train_sen, train_auc, test_loss, test_acc, test_spec, test_sen, test_auc]
        data = pd.DataFrame([data_csv])
        data.to_csv(log_path, mode='a', header=False, index=False)
        # if epoch % 5 == 0:
        #     print('Saving epoch %d model ...' % (epoch + 1))
        #     state = {
        #         'net': model.state_dict(),
        #         'epoch': epoch + 1
        #     }
        #     torch.save(state, model_path + '/' + 'CP_epoch_%d_%.3f.ckpt' % (epoch + 1, test_auc))

        if test_auc >= 0.80 and best_auc < test_auc:
            best_auc = test_auc
            min_loss = test_loss
            print('Saving epoch %d model ...' % (epoch + 1))
            state = {
                'net': model.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(state, model_path + '/' + 'best_CP_model_%d_%.2f_%.2f.ckpt' % (epoch + 1, test_auc, test_loss))

        # Save the best model by loss
        if test_loss < min_loss:
            min_loss = test_loss
            print('Saving epoch %d model with best Loss (%.3f) ...' % (epoch + 1, test_loss))
            state = {
                'net': model.state_dict(),
                'epoch': epoch + 1
            }
            torch.save(state, model_path + '/' + 'best_CP_model_%d_%.2f_%.2f.ckpt' % (epoch + 1, test_auc, test_loss))
