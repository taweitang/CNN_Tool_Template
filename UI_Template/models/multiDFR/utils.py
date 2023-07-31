import numpy as np
import os
import cv2
from skimage.io import imread, imsave, imshow
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



def normalize(x):
    """ Normalize x to [0, 1]
    """
    x_min = x.min()
    x_max = x.max()
    return (x - x_min) / (x_max - x_min)


def visulization(img_file, mask_path, score_map_path, saving_path):
    # image name
    img_name = img_file.split("/")
    img_name = "-".join(img_name[-2:])

    # image
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    #     imsave("feature_maps/Results/gt_image/{}".format(img_name), image)

    # mask
    mask_file = os.path.join(mask_path, img_name)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), 2)

    # binary score {0, 255}
    score_file = os.path.join(score_map_path, img_name)
    score = cv2.imread(score_file, cv2.IMREAD_GRAYSCALE)
    img = img[:, :, ::-1]  # bgr to rgb
    img[..., 1] = np.where(score == 255, 255, img[..., 1])

    # save
    imsave(os.path.join(saving_path, "{}".format(img_name)), img)

def visulization_score(img_file, mask_path, score_map_path, saving_path):
    # image name
    img_name = img_file.split("/")
    img_name = "-".join(img_name[-2:])

    # image
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (256, 256))
    #     imsave("feature_maps/Results/gt_image/{}".format(img_name), image)

    superimposed_img = img.copy()

    # mask
    mask_file = os.path.join(mask_path, img_name)
    mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), thickness=-1)
    img = img[:, :, ::-1]  # bgr to rgb

    # normalized score {0, 255}
    score_file = os.path.join(score_map_path, img_name)
    score = cv2.imread(score_file, cv2.IMREAD_GRAYSCALE)

    heatmap = cv2.applyColorMap(score, cv2.COLORMAP_JET)  # 将score转换成热力图
    superimposed_img = heatmap * 0.7 + superimposed_img * 0.8     # 将热力图叠加到原图像
    # cv2.imwrite('cam.jpg', superimposed_img)  # 将图像保存

    # save
    cv2.imwrite(os.path.join(saving_path, "{}".format(img_name)), superimposed_img)
    imsave(os.path.join(saving_path, "gt_{}".format(img_name)), img)


def spec_sensi_acc_iou_auc(mask, binary_score, score):
    """
    ref: iou https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    ref: confusion matrix https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    ref: confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    binary_score[binary_score > 0.5] = 1
    binary_score[binary_score <= 0.5] = 0

    gt_n = mask == 0
    pred_n = binary_score == 0
    gt_p = mask == 1
    pred_p = binary_score == 1

    specificity = np.sum(gt_n * pred_n) / np.sum(gt_n)
    sensitivity = np.sum(gt_p * pred_p) / np.sum(gt_p)
    accuracy = (np.sum(gt_p * pred_p) + np.sum(gt_n * pred_n)) / (np.sum(gt_p) + np.sum(gt_n))
    # coverage = np.sum(score * mask) / (np.sum(score) + np.sum(mask))

    intersection = np.logical_and(mask, binary_score)
    union = np.logical_or(mask, binary_score)
    iou_score = np.sum(intersection) / np.sum(union)

    auc_score = roc_auc_score(mask.ravel(), score.ravel())

    return specificity, sensitivity, accuracy, iou_score, auc_score


def spec_sensi_acc_riou_auc(mask, binary_score, score):
    """
    ref: iou https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    ref: confusion matrix https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    ref: confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    binary_score[binary_score > 0.5] = 1
    binary_score[binary_score <= 0.5] = 0

    gt_n = mask == 0
    pred_n = binary_score == 0
    gt_p = mask == 1
    pred_p = binary_score == 1

    specificity = np.sum(gt_n * pred_n) / np.sum(gt_n)      # recall for negtive
    # specificity = np.sum(gt_p * pred_p) / np.sum(pred_p)    # precision
    sensitivity = np.sum(gt_p * pred_p) / np.sum(gt_p)      # recall for positive
    accuracy = (np.sum(gt_p * pred_p) + np.sum(gt_n * pred_n)) / (np.sum(gt_p) + np.sum(gt_n))
    # coverage = np.sum(score * mask) / (np.sum(score) + np.sum(mask))

    intersection = np.logical_and(mask, binary_score)
    union = np.logical_or(mask, binary_score)
    # iou_score = np.sum(intersection) / np.sum(union)
    iou_score = np.sum(intersection) / np.sum(mask)    # relative iou

    auc_score = roc_auc_score(mask.ravel(), score.ravel())

    fpr, tpr, thresholds = roc_curve(mask.ravel(), score.ravel(), pos_label=1)

    return specificity, sensitivity, accuracy, iou_score, auc_score, [fpr, tpr, thresholds]


def auc_roc(mask, score):
    """
    ref: iou https://www.jeremyjordan.me/evaluating-image-segmentation-models/
    ref: confusion matrix https://stackoverflow.com/questions/31324218/scikit-learn-how-to-obtain-true-positive-true-negative-false-positive-and-fal
    ref: confusion matrix https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0

    auc_score = roc_auc_score(mask.ravel(), score.ravel())
    fpr, tpr, thresholds = roc_curve(mask.ravel(), score.ravel(), pos_label=1)

    return auc_score, [fpr, tpr, thresholds]


def rescale(x):
    return (x - x.min()) / (x.max() - x.min())


# 3d roc
def AUCs(y_test,score,mask=False):

    if mask:
        PF, PD, thresholds = roc_curve(y_test.ravel(), score.ravel(), pos_label=1)
    else:
        PF, PD, thresholds = roc_curve(y_test, score, pos_label=1)

    PF = np.delete(PF, [0])
    PD = np.delete(PD,[0])
    thresholds = np.delete(thresholds,[0])
    thresholds = (thresholds-min(thresholds))/(max(thresholds)-min(thresholds))
    AUC = 0
    AUCDT = 0
    AUCFT = 0
    for i in range (1,len(thresholds)):
        area =  PD[i] * ( PF[i] - PF[i-1])
        areaDT = (thresholds[i-1]-thresholds[i]) * PD[i]
        areaFT = (thresholds[i-1]-thresholds[i]) * PF[i]
        
        AUC += area
        AUCDT+=areaDT
        AUCFT+=areaFT


    AUCTD = AUCDT
    AUCBS = AUCFT
    AUCTDBS = AUCDT - AUCFT
    
    return AUC,AUCTD,AUCBS,AUCTDBS,[PF, PD, thresholds]

def Draw3DROC(TDROC,path):
    PF, PD, thresholds = TDROC[0], TDROC[1], TDROC[2]
    thresholds = np.concatenate(([1], thresholds, [0]))
    PF = np.concatenate(([0], PF, [1]))
    PD = np.concatenate(([0], PD, [1]))

    fig = plt.figure()
    ax = Axes3D(fig)
    # For line plot
    ax.plot(thresholds, PF, PD, c='r')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('False positive rate')
    ax.set_zlabel('True positive rate')
    plt.savefig(path)

