import torch
import matplotlib.pyplot as plt
import numpy as np
from model.ResNet18 import CP_ResNet18_mutli_attention
from Data_loader import ReadDataFromCSV
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import os
import pandas as pd
import random

seed_value = 42  # set random seed
np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # disable hash randomization for reproducibility
torch.manual_seed(seed_value)  # set seed for CPU
torch.cuda.manual_seed(seed_value)  # set seed for the current GPU (single GPU)
torch.cuda.manual_seed_all(seed_value)  # set seed for all GPUs (multi-GPU)
torch.backends.cudnn.deterministic = True


def test(data_loader, model, type):
    predictions = []  # model predictions
    # paths = []  # path information of each sample
    target = []
    preds = []
    total = 0
    correct = 0
    true_negatives = 0
    false_positives = 0
    true_positives = 0
    false_negatives = 0
    incorrect_samples = []  # paths or indices of misclassified samples
    probabilities = []  # predicted class probabilities
    names = []
    with torch.no_grad():
        for batch_idx_CT, data in enumerate(data_loader):
            images_CT, images_PET, labels_T, labels_N, name = data  # CT/PET data and labels
            # names.append(name)
            names.extend(name)
            # Make sure CT and PET labels match (depends on your task and data)
            # assert len(images_CT) == len(images_PET), "Labels for CT and PET do not match."
            # Move data to GPU
            images_CT, images_PET, labels_T, labels_N = images_CT.to(device), images_PET.to(device), labels_T.to(device), labels_N.to(device)
            outputs = model(images_CT, images_PET)
            # outputs = model(images_CT)

            labels = labels_T

            # sigmoid
            outputs2 = torch.sigmoid(outputs)

            # probabilities.extend(outputs2.detach().cpu().numpy())  # move predictions to CPU as NumPy arrays
            probabilities.extend(outputs2.flatten().cpu().numpy().tolist())

            predicted = torch.where(outputs2 >= 0.5, torch.tensor(1, device=device), torch.tensor(0, device=device))

            # preds.extend(predicted.detach().cpu().numpy())
            preds.extend(predicted.flatten().cpu().numpy().tolist())

            # Ground-truth labels
            # target.extend(labels.cpu().numpy())
            target.extend(labels.flatten().cpu().numpy().tolist())

            data_csv = [name[0], labels.detach().cpu().numpy()[0], type, outputs2.detach().cpu().numpy()[0][0]]
            data = pd.DataFrame([data_csv])
            data.to_csv(log_path, mode='a', header=False, index=False)

            # # Predictions and targets
            # _, predicted = torch.max(outputs, 1)
            # probability = F.softmax(outputs, dim=1)
            # probabilities.extend(probability[:, 1].detach().cpu().numpy())
            # pred.extend(predicted.detach().cpu().numpy())

            predicted = predicted.view(-1)
            labels = labels.view(-1)

            total += labels.size(0)
            # # print(total)
            correct += (predicted == labels).sum().item()

            # Update true negatives, false positives, true positives and false negatives
            true_negatives += ((predicted == 0) & (labels == 0)).sum().item()  # TN: predicted 0, actual 0
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()  # FP: predicted 1, actual 0
            true_positives += ((predicted == 1) & (labels == 1)).sum().item()  # TP: predicted 1, actual 1
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()  # FN: predicted 0, actual 1

        print(true_negatives, false_positives, true_positives, false_negatives)
        # Compute accuracy, precision, recall, F1-score, AUC, etc.
        # Confusion matrix
        cm = confusion_matrix(target, preds)
        accuracy = correct / total
        precision = precision_score(target, preds)
        recall = recall_score(target, preds)  # recall (sensitivity)
        f1 = f1_score(target, preds)
        fpr, tpr, threshold = roc_curve(target, probabilities)
        roc_auc = auc(fpr, tpr)
        # Find the optimal threshold
        optimal_idx = np.argmax(tpr - fpr)  # maximize TPR - FPR
        optimal_threshold = threshold[optimal_idx]
        print(cm)
        print('Optimal threshold:', optimal_threshold)

        # Specificity and sensitivity
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        # specificity = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (true_negatives + false_positives) > 0 else 0
        # sensitivity = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (true_positives + false_negatives) > 0 else 0

    print(f"{type} metrics:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Specificity: {specificity:.2f}")  # print specificity
    print(f"Sensitivity: {sensitivity:.2f}")  # print sensitivity
    # print(f"Precision: {precision:.2f}")
    # print(f"Recall: {recall:.2f}")  # recall is already computed as sensitivity
    print(f"F1-score: {f1:.2f}")
    print(f"AUC: {roc_auc:.2f}")
    # print(names)
    # print(target)
    # probs = [prob[0] for prob in probabilities]
    # pred = [pred[0] for pred in preds]
    # print(probs)
    # print(pred)

    return fpr, tpr, roc_auc

if __name__ == '__main__':
    # Check whether a GPU device is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device:', device)

    # log_path = './result/result_train.csv'
    log_path = './result/log/result.csv'

    # Create the log directory if it does not exist
    log_dir = os.path.dirname(log_path)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    # Create an empty log file if it does not exist
    if not os.path.isfile(log_path):
        open(log_path, 'w').close()
    df = pd.DataFrame(columns=['name', 'label', 'group', 'Model'])  # column names
    df['label'] = df['label'].astype('U')  # convert the 'label' column to Unicode strings
    df.to_csv(log_path, index=False)  # path can be changed as needed

    # model = CT_ResNet18(1, 1)
    model = CP_ResNet18_mutli_attention(1, 1)
    model.to(device)
    # Load an existing model
    checkpoint = torch.load('./result/checkpoint/best_model.ckpt', map_location=torch.device('cuda:0'), weights_only=True)
    model.load_state_dict(checkpoint['net'])
    epoch = checkpoint['epoch']
    model.eval()
    batch_size = 16
    train_csv = r'./Dataset/train/train.xlsx'
    test_csv = r'./Dataset/test/test.xlsx'
    train_data = pd.read_excel(train_csv)
    test_data = pd.read_excel(test_csv)

    trainset = ReadDataFromCSV(train_csv, transform=None, Data_Aug=0, mode='CP', fusion='late')
    testset = ReadDataFromCSV(test_csv, transform=None, Data_Aug=0, mode='CP', fusion='late')

    train_loader = DataLoader(trainset, batch_size, shuffle=False, num_workers=3, pin_memory=True)
    # Do not use WeightedRandomSampler here
    test_loader = DataLoader(testset, batch_size, shuffle=False, num_workers=3, pin_memory=True)
    train_fpr, train_tpr, train_roc = test(train_loader, model, 'train')
    test_fpr, test_tpr, test_roc = test(test_loader, model, 'test')

    # Plot ROC curves
    plt.figure()
    plt.plot(train_fpr, train_tpr, color='blue', lw=2, label='Training AUC: %0.2f' % train_roc)
    plt.plot(test_fpr, test_tpr, color='orange', lw=2, label='Testing AUC: %0.2f' % test_roc)
    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")

    # plt.savefig('./result/log/ROC.jpg')
    plt.savefig('./result/log/ROC.jpg')

    plt.show()
