from __future__ import print_function
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from sklearn import metrics
import numpy
import logging
import importlib
import os
from fl_enum import PackageLogMsg,LogLevel

import json


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


if __name__ == "__main__":
    print("validation started ...")

    progress = {
        "status": "initialization",
        "completedPercentage": 0
    }
    with open('/var/output/progress.json', 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)

    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args, unparsed = parser.parse_known_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    test_kwargs = {'batch_size': args.test_batch_size}

    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        test_kwargs.update(cuda_kwargs)

    progress = {
        "status": "preprocessing",
        "completedPercentage": 0
    }
    with open('/var/output/progress.json', 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)


    dataset = None
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    try:
        dataset = datasets.MNIST('/data', train=False, download=False, transform=transform)
    except Exception as err:
        with open('/var/logs/error.log', 'a') as fd:
            fd.write(f"load dataset failed: " + str(err))
        os._exit(os.EX_OK)

    test_loader = torch.utils.data.DataLoader(dataset, **test_kwargs)
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # load testing model weight
    try:
        model.load_state_dict(torch.load("/var/model/merge.ckpt")["state_dict"])
    except Exception as err:
        with open('/var/logs/error.log', 'a') as fd:
            fd.write(f"load model failed: " + str(err))
        os._exit(os.EX_OK)

    model.eval()
    y_pred = []
    y_probobility = []
    y_true = []


    progress = {
        "status": "validating",
        "completedPercentage": 0
    }
    with open('/var/output/progress.json', 'w', encoding='utf-8') as f:
        json.dump(progress, f, ensure_ascii=False, indent=4)


    with torch.no_grad():
        try:
            for data, labels in test_loader:
                data, labels = data.to(device), labels.to(device)
                output = model(data)
                pred_list = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                y_pred.extend(pred_list) # Save Prediction

                labels = labels.data.cpu().numpy()
                y_true.extend(labels) # Save Truth

                # Saving probobilities for roc curve
                for i in range(len(pred_list)):
                    pred = pred_list[i]
                    y_probobility.append(output[i][pred].cpu().numpy())

        except Exception as err:
            with open('/var/logs/error.log', 'a') as fd:
                fd.write(f"validating failed: " + str(err))
            os._exit(os.EX_OK)

        general_confusion_matrix = [[0,0],[0,0]]
        precision_list = []
        recall_list = []
        f1_score_list = []

        cf_matrix = metrics.multilabel_confusion_matrix(y_true, y_pred)
        g_matrix = numpy.nan_to_num(metrics.confusion_matrix(y_true, y_pred))
        for matrix in cf_matrix:
            precision = matrix[1][1] / (matrix[0][1]+matrix[1][1])
            recall = matrix[1][1] / (matrix[1][1] + matrix[1][0])
            f1_score = 2 * ( precision * recall ) / ( precision + recall )

            precision_list.append(numpy.nan_to_num(precision))
            recall_list.append(numpy.nan_to_num(recall))
            f1_score_list.append(numpy.nan_to_num(f1_score))

            general_confusion_matrix[0][0] = general_confusion_matrix[0][0] + matrix[0][0]  # TN = True Negative
            general_confusion_matrix[0][1] = general_confusion_matrix[0][1] + matrix[0][1]  # FP = False Positive
            general_confusion_matrix[1][0] = general_confusion_matrix[1][0] + matrix[1][0]  # FN = False Negative
            general_confusion_matrix[1][1] = general_confusion_matrix[1][1] + matrix[1][1]  # TP = True Positive

        general_precision = numpy.nan_to_num((general_confusion_matrix[1][1] / (general_confusion_matrix[0][1]+general_confusion_matrix[1][1])))
        general_recall = numpy.nan_to_num(general_confusion_matrix[1][1] / (general_confusion_matrix[1][1] + general_confusion_matrix[1][0]))
        general_f1_score = numpy.nan_to_num(2 * ( precision * recall ) / ( precision + recall ))


        fpr_list = []
        tpr_list = []
        for i in range(10):
            fpr, tpr, thresholds = metrics.roc_curve(y_pred, y_probobility, pos_label=i)
            fpr_list.append(numpy.nan_to_num(fpr))
            tpr_list.append(numpy.nan_to_num(tpr))

        progress = {
            "status": "completed",
            "completedPercentage": 100
        }
        with open('/var/output/progress.json', 'w', encoding='utf-8') as f:
            json.dump(progress, f, ensure_ascii=False, indent=4)

        result = {
            "metadata": {
                "datasetSize": len(test_loader.dataset)*10,
            },
            "results": {
                "tables": [
                    {
                        "title": "average evaluation metrics",
                        "labels":["f1","precision","recall"],
                        "values": [str(general_f1_score),str(general_precision),str(general_recall)]
                    },
                    {
                        "title": "Number 0 evaluation metrics",
                        "cols": ["f1", "precision", "recall"],
                        "values": [str(f1_score_list[0]),str(precision_list[0]),str(recall_list[0])]
                    },
                    {
                        "title": "Number 1 evaluation metrics",
                        "cols": ["f1", "precision", "recall"],
                        "values": [str(f1_score_list[1]),str(precision_list[1]),str(recall_list[1])]
                    },
                    {
                        "title": "Number 2 evaluation metrics",
                        "cols": ["f1", "precision", "recall"],
                        "values": [str(f1_score_list[2]),str(precision_list[2]),str(recall_list[2])]
                    },
                    {
                        "title": "Number 3 evaluation metrics",
                        "cols": ["f1", "precision", "recall"],
                        "values": [str(f1_score_list[3]),str(precision_list[3]),str(recall_list[3])]
                    },
                    {
                        "title": "Number 4 evaluation metrics",
                        "cols": ["f1", "precision", "recall"],
                        "values": [str(f1_score_list[4]),str(precision_list[4]),str(recall_list[4])]
                    },
                    {
                        "title": "Number 5 evaluation metrics",
                        "cols": ["f1", "precision", "recall"],
                        "values": [str(f1_score_list[5]),str(precision_list[5]),str(recall_list[5])]
                    },
                    {
                        "title": "Number 6 evaluation metrics",
                        "cols": ["f1", "precision", "recall"],
                        "values": [str(f1_score_list[6]),str(precision_list[6]),str(recall_list[6])]
                    },
                    {
                        "title": "Number 7 evaluation metrics",
                        "cols": ["f1", "precision", "recall"],
                        "values": [str(f1_score_list[7]),str(precision_list[7]),str(recall_list[7])]
                    },
                    {
                        "title": "Number 8 evaluation metrics",
                        "cols": ["f1", "precision", "recall"],
                        "values": [str(f1_score_list[8]),str(precision_list[8]),str(recall_list[8])]
                    },
                    {
                        "title": "Number 9 evaluation metrics",
                        "cols": ["f1", "precision", "recall"],
                        "values": [str(f1_score_list[9]),str(precision_list[9]),str(recall_list[9])]
                    },
                ],
                "heatmaps":[
                    {
                        "title": "10 numbers 's ConfusionMatrix",
                        "x-labels":["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                        "y-labels":["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
                        "x-axis":"numbers",
                        "y-axis":"numbers",
                        "values":g_matrix.tolist(),
                    },
                ],
                "plots": [
                    {
                        "title": "Number 0 's roc curve",
                        "labels":["Number 0"],
                        "x-axis":"fpr",
                        "y-axis":"tpr",
                        "x-values":[[fpr_list[0].tolist()]],
                        "y-values":[[tpr_list[0].tolist()]],
                    },
                    {
                        "title": "Number 1 's roc curve",
                        "labels":["Number 1"],
                        "x-axis":"fpr",
                        "y-axis":"tpr",
                        "x-values":[[fpr_list[1].tolist()]],
                        "y-values":[[tpr_list[1].tolist()]],
                    },
                    {
                        "title": "Number 2 's roc curve",
                        "labels":["Number 2"],
                        "x-axis":"fpr",
                        "y-axis":"tpr",
                        "x-values":[[fpr_list[2].tolist()]],
                        "y-values":[[tpr_list[2].tolist()]],
                    },
                    {
                        "title": "Number 3 's roc curve",
                        "labels":["Number 3"],
                        "x-axis":"fpr",
                        "y-axis":"tpr",
                        "x-values":[[fpr_list[3].tolist()]],
                        "y-values":[[tpr_list[3].tolist()]],
                    },
                    {
                        "title": "Number 4 's roc curve",
                        "labels":["Number 4"],
                        "x-axis":"fpr",
                        "y-axis":"tpr",
                        "x-values":[[fpr_list[4].tolist()]],
                        "y-values":[[tpr_list[4].tolist()]],
                    },
                    {
                        "title": "Number 5 's roc curve",
                        "labels":["Number 5"],
                        "x-axis":"fpr",
                        "y-axis":"tpr",
                        "x-values":[[fpr_list[5].tolist()]],
                        "y-values":[[tpr_list[5].tolist()]],
                    },
                    {
                        "title": "Number 6 's roc curve",
                        "labels":["Number 6"],
                        "x-axis":"fpr",
                        "y-axis":"tpr",
                        "x-values":[[fpr_list[6].tolist()]],
                        "y-values":[[tpr_list[6].tolist()]],
                    },
                    {
                        "title": "Number 7 's roc curve",
                        "labels":["Number 7"],
                        "x-axis":"fpr",
                        "y-axis":"tpr",
                        "x-values":[[fpr_list[7].tolist()]],
                        "y-values":[[tpr_list[7].tolist()]],
                    },
                    {
                        "title": "Number 8 's roc curve",
                        "labels":["Number 8"],
                        "x-axis":"fpr",
                        "y-axis":"tpr",
                        "x-values":[[fpr_list[8].tolist()]],
                        "y-values":[[tpr_list[8].tolist()]],
                    },
                    {
                        "title": "Number 9 's roc curve",
                        "labels":["Number 9"],
                        "x-axis":"fpr",
                        "y-axis":"tpr",
                        "x-values":[[fpr_list[9].tolist()]],
                        "y-values":[[tpr_list[9].tolist()]],
                    },
                ],
            },
        }

        with open('/var/output/result.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=4)



