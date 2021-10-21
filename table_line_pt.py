#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 9 23:11:51 2020
table line detect
@author: chineseocr
"""

import torch
from models.model import table_net

from utils import letterbox_image, get_table_line, adjust_lines, line_to_line
import numpy as np
import cv2

def table_line(img, model, size=(512, 512), hprob=0.5, vprob=0.5, row=50, col=30, alph=15):
    sizew, sizeh = size
    inputBlob, fx, fy = letterbox_image(img[..., ::-1], (sizew, sizeh))

    # pred = model.predict(np.array([np.array(inputBlob) / 255.0]))
    pred = model(torch.FloatTensor(np.array([np.array(inputBlob) / 255.0])).permute(0, 3, 1, 2))

    pred = pred.permute(0, 2, 3, 1)
    pred = pred.detach().numpy()
    pred = pred[0]
    vpred = pred[..., 1] > vprob  ##竖线
    hpred = pred[..., 0] > hprob  ##横线
    vpred = vpred.astype(int)
    hpred = hpred.astype(int)
    colboxes = get_table_line(vpred, axis=1, lineW=col)
    rowboxes = get_table_line(hpred, axis=0, lineW=row)
    ccolbox = []
    crowlbox = []
    if len(rowboxes) > 0:
        rowboxes = np.array(rowboxes)
        rowboxes[:, [0, 2]] = rowboxes[:, [0, 2]] / fx
        rowboxes[:, [1, 3]] = rowboxes[:, [1, 3]] / fy
        xmin = rowboxes[:, [0, 2]].min()
        xmax = rowboxes[:, [0, 2]].max()
        ymin = rowboxes[:, [1, 3]].min()
        ymax = rowboxes[:, [1, 3]].max()
        ccolbox = [[xmin, ymin, xmin, ymax], [xmax, ymin, xmax, ymax]]
        rowboxes = rowboxes.tolist()

    if len(colboxes) > 0:
        colboxes = np.array(colboxes)
        colboxes[:, [0, 2]] = colboxes[:, [0, 2]] / fx
        colboxes[:, [1, 3]] = colboxes[:, [1, 3]] / fy

        xmin = colboxes[:, [0, 2]].min()
        xmax = colboxes[:, [0, 2]].max()
        ymin = colboxes[:, [1, 3]].min()
        ymax = colboxes[:, [1, 3]].max()
        colboxes = colboxes.tolist()
        crowlbox = [[xmin, ymin, xmax, ymin], [xmin, ymax, xmax, ymax]]

    # rowboxes += crowlbox
    # colboxes += ccolbox

    rboxes_row_, rboxes_col_ = adjust_lines(rowboxes, colboxes, alph=alph)
    rowboxes += rboxes_row_
    colboxes += rboxes_col_
    nrow = len(rowboxes)
    ncol = len(colboxes)
    for i in range(nrow):
        for j in range(ncol):
            rowboxes[i] = line_to_line(rowboxes[i], colboxes[j], 10)
            colboxes[j] = line_to_line(colboxes[j], rowboxes[i], 10)

    return rowboxes, colboxes


if __name__ == '__main__':
    import time

    # p = 'img/table-detect.jpg'
    p = 'img/4.tif'
    model_path = 'xx.pt'

    model = table_net(2)
    # model.load_weights(tableModeLinePath)
    model.load_state_dict(torch.load('./table_line_pt.pt'))
    model.eval()

    from utils import draw_lines

    img = cv2.imread(p)
    t = time.time()
    rowboxes, colboxes = table_line(img[..., ::-1], model, size=(512, 512), hprob=0.3, vprob=0.3)
    img = draw_lines(img, rowboxes + colboxes, color=(255, 0, 0), lineW=2)

    print(time.time() - t, len(rowboxes), len(colboxes))
    cv2.imwrite('img/table-line.png', img)
