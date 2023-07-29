import numpy as np
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from .common import ci95


def roc_and_auc(label_set: list, predict_set: list, tag_set: list, n_class: int, ylabel: str="TPR", xlabel: str="FPR"):
    # [label1, label2]
    assert len(label_set) == len(predict_set) == len(tag_set)
    plt_figure = plt.figure()
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    color_set = ["orangered", "cyan", "fuchsia", "darkorange", "mediumseagreen", "cornflowerblue", "darkorchid"]
    for plt_index in range(len(label_set)):
        assert len(label_set[plt_index]) == len(predict_set[plt_index])
        fpr_list = []
        tpr_list = []
        for index in range(n_class):
            one_hot_set = label_binarize(label_set[plt_index], classes=[i for i in range(n_class)])
            normalized_predict_set = predict_set[plt_index][:, index]
            fpr, tpr, threshold = roc_curve(one_hot_set[:, index], normalized_predict_set)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
            for i in range(len(tpr)):
                if i == 0:
                    ks_max = tpr[i] - fpr[i]
                    best_thr = threshold[i]
                elif tpr[i] - fpr[i] > ks_max:
                    ks_max = tpr[i] - fpr[i]
                    best_thr = threshold[i]
            print("{} 类别{} AUC为{} 阈值为{}".format(tag_set[plt_index], index, auc(fpr, tpr), best_thr))
        final_fpr = np.unique(np.concatenate([fpr_list[index] for index in range(n_class)]))
        final_tpr = np.zeros_like(final_fpr)
        for index in range(n_class):
            final_tpr += np.interp(final_fpr, fpr_list[index], tpr_list[index])
        final_tpr /= n_class
        final_auc = auc(final_fpr, final_tpr)
        plt.plot(final_fpr, final_tpr, color=color_set[plt_index], label="{}, AUC={:.2f}".format(tag_set[plt_index], final_auc))
    plt.legend()
    return plt_figure

def roc_and_aucv2(label_set: list, predict_set: list, tag_set: list, n_class: int, ylabel: str="TPR", xlabel: str="FPR", head="", filename=""):
    # [label1, label2]
    plt.cla()
    plt.figure(dpi=300)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    color_set = ["orangered", "cyan", "fuchsia", "darkorange", "mediumseagreen", "cornflowerblue", "darkorchid"]
    for index in range(n_class):
        one_hot_set = label_binarize(label_set, classes=[i for i in range(n_class)])
        if n_class == 2:
            one_hot_set = np.hstack((1 - one_hot_set, one_hot_set))
        fpr, tpr, threshold = roc_curve(one_hot_set[:, index], predict_set[:, index])
        auc_value = auc(fpr, tpr)
        auc_95_list = []
        for auc_95_index in range(8):
            auc_95_fpr, auc_95_tpr, _ = roc_curve(one_hot_set[auc_95_index * 1000:(auc_95_index+1) * 1000, index], predict_set[auc_95_index * 1000:(auc_95_index+1) * 1000, index])
            auc_95_list.append(auc(auc_95_fpr, auc_95_tpr))
        auc_95_value = ci95(np.array(auc_95_list))
        auc_95_value = [min(max(item, 0), 1) for item in auc_95_value]
        print("类别:{} | AUC为{} | 95%CI为{}".format(tag_set[index], auc_value, auc_95_value))
        plt.plot(fpr, tpr, color=color_set[index], label="{}, AUC={:.4f}".format(tag_set[index], auc_value))
    plt.legend(loc=0)
    plt.savefig(filename)

def cm(c_matrix, class_list, figure_name="cm_smallZuwai.png"):
    plt.cla()
    c_matrix = c_matrix.astype(np.float)
    assert len(class_list) == c_matrix.shape[0] == c_matrix.shape[1]
    cmap = plt.cm.get_cmap('Blues')
    plt.figsize=(10, 10)
    plt.imshow(c_matrix, cmap=cmap)
    plt.colorbar()
    xlocations = np.array(range(len(class_list)))
    plt.xticks(xlocations, class_list, rotation=60)
    plt.yticks(xlocations, class_list)
    plt.xlabel('Predict label')
    plt.ylabel('True label')
    plt.title('Confusion_Matrix')
    for i in range(c_matrix.shape[0]):
        for j in range(c_matrix.shape[1]):
            plt.text(x=j, y=i, s=int(c_matrix[i, j]), va='center', ha='center', color='black', fontsize=10)
    plt.savefig(figure_name,dpi=500)