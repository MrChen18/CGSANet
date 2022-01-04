import numpy as np
from datetime import datetime

def show_cd_rpqf1_pixel(hist):
    """
                     GT:Changed, Unchanged
      Predicted-Changed:  TP   ,     FP    , b1
    Predicted-Unchanged:  FN   ,     TN    , b2
                          a1   ,     a2
    """
    tp = hist[0,0]
    tn = hist[1,1]
    fp = hist[0,1]# gt->0,predict->1, false alarm
    fn = hist[1,0]# gt->1,predict->0, missed detection
    all_pixel_count = tp + tn + fp + fn
    # changed_pixel_count = tp + fn
    # unchanged_pixel_count = all_pixel_count - changed_pixel_count#fp,tn
    if tp == 0:
        recall = 0
        precision = 0
        f1measure = 0
    else:
        recall = tp * 100.0 / (tp + fn)
        precision = tp * 100.0 / (tp + fp)
        f1measure = 2.0 * recall * precision / (recall + precision)
    # if fn == 0: 
    #     misdetection = 0
    # else:
    #     misdetection = fn * 100.0 / changed_pixel_count #lou bao#fn / tp,fn
    # if fp == 0: 
    #     falsealarms = 0
    # else:    
    #     falsealarms = fp * 100.0 / unchanged_pixel_count #wu bao,#fp / fp,tn
        
    # totalerror = (fp + fn) * 100.0 / all_pixel_count
    accuray = (tp + tn) * 100.0 / all_pixel_count
    # pra = (tp + tn) * 1.0 / all_pixel_count
    # a1 = tp + fn #changed_pixel_count
    # b1 = tp + fp
    # a2 = fp + tn #unchanged_pixel_count
    # b2 = fn + tn
    # pre = (a1 * b1 + a2 * b2) * 1.0 / (all_pixel_count * all_pixel_count)
    # kappa = (pra - pre) / (1 - pre)
    
    iou = tp * 100.0 / (tp + fp + fn)
    print( datetime.now())
    print("------accuracy-------")
    # print(" False Alarms:{0:.2f}%".format(falsealarms))
    # print(" Misdetection:{0:.2f}%".format(misdetection))
    print("       Recall:{0:.2f}%".format(recall))
    print("    Precision:{0:.2f}%".format(precision))
    print("          IoU:{0:.2f}%".format(iou))
    print("     F1-Score:{0:.2f}%".format(f1measure))    
    print("      Accuray:{0:.2f}%".format(accuray))
    # print("Overall error:{0:.2f}%".format(totalerror))
    # print("        Kappa:{0:.4f}".format(kappa))
    print("--------------------")
    # return recall, precision, f1measure, iou, accuray, kappa
    return recall, precision, iou, f1measure, accuray

def hist(gt_data,pre_data):
    gt_data[gt_data > 0.5] = 1
    gt_data[gt_data < 1] = 0
    pre_data[pre_data > 0.5] = 1
    pre_data[pre_data < 1] = 0
    hist = np.zeros((2, 2))
    #tp
    tp = np.count_nonzero((gt_data == pre_data) & (gt_data > 0))
    #tn 
    tn = np.count_nonzero((gt_data == pre_data) & (gt_data ==  0))
    #fp
    fp = np.count_nonzero(gt_data < pre_data)
    #fn
    fn =  np.count_nonzero(gt_data > pre_data)
    hist[0,0] = hist[0,0] + tp
    hist[1,1] = hist[1,1] + tn
    hist[0,1] = hist[0,1] + fp
    hist[1,0] = hist[1,0] + fn
    return hist