import os
import argparse
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import f1_score
from utils import AverageMeter
from metrics import dice_coef, iou_score,get_F1,get_accuracy,get_recall,get_precision
if __name__=='__main__':
    parse=argparse.ArgumentParser()
    parse.add_argument('--testpath',type=str,default="")
    parse.add_argument('--path',type=str,default="")

    args=parse.parse_args()

    dataname=os.listdir(args.path)

    #for _dataname in dataname:
    for _dataname in ['']:
        gtpath=os.path.join(args.path,_dataname,"masks")
        prepath=os.path.join(args.testpath,_dataname)
        files=os.listdir(gtpath)
        lens=len(files)
        print("##"*20)
        print(_dataname,"length: ",lens)
        print("##"*20)
        iou= AverageMeter()
        dice= AverageMeter()
        f1=AverageMeter()
        acc=AverageMeter()
        recall=AverageMeter()
        precision=AverageMeter()
        for file in files:
            gtfile=os.path.join(gtpath,file)
            prefile=os.path.join(prepath,file)
            gt=Image.open(gtfile).convert('L')
            pre=Image.open(prefile).convert('L')
            gt=np.asarray(gt)
            pre=np.asarray(pre)
            h,w=gt.shape[:2]
            gt=gt.reshape(h,w,1)/255.0
            pre=pre.reshape(h,w,1)/255.0
            iou.update(iou_score(pre,gt))
            dice.update(dice_coef(pre,gt))
            f1.update(get_F1(pre,gt))
            acc.update(get_accuracy(pre,gt))
            recall.update(get_recall(pre,gt))
            precision.update(get_precision(pre,gt))
        print("acc:",acc.avg)
        print("Iou",iou.avg)
        print("Dice",dice.avg)
        print("f1",f1.avg)
        print("recall",recall.avg)
        print("precision",precision.avg)