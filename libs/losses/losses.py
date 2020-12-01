# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from libs.box_utils import bbox_transform
from libs.box_utils.iou_rotate import iou_rotate_calculate2
from libs.configs import cfgs
import numpy as np


def focal_loss_(labels, pred, anchor_state, alpha=0.25, gamma=2.0):

    # filter out "ignore" anchors
    indices = tf.reshape(tf.where(tf.not_equal(anchor_state, -1)), [-1, ])
    labels = tf.gather(labels, indices)
    pred = tf.gather(pred, indices)

    logits = tf.cast(pred, tf.float32)
    onehot_labels = tf.cast(labels, tf.float32)
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=onehot_labels, logits=logits)
    predictions = tf.sigmoid(logits)
    predictions_pt = tf.where(tf.equal(onehot_labels, 1), predictions, 1.-predictions)
    alpha_t = tf.scalar_mul(alpha, tf.ones_like(onehot_labels, dtype=tf.float32))
    alpha_t = tf.where(tf.equal(onehot_labels, 1.0), alpha_t, 1-alpha_t)
    loss = ce * tf.pow(1-predictions_pt, gamma) * alpha_t
    positive_mask = tf.cast(tf.greater(labels, 0), tf.float32)
    return tf.reduce_sum(loss) / tf.maximum(tf.reduce_sum(positive_mask), 1)


def focal_loss(labels, pred, anchor_state, alpha=0.25, gamma=2.0):

    # filter out "ignore" anchors
    indices = tf.reshape(tf.where(tf.not_equal(anchor_state, -1)), [-1, ])
    labels = tf.gather(labels, indices)
    pred = tf.gather(pred, indices)

    # compute the focal loss
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=labels, logits=pred))
    prediction_probabilities = tf.sigmoid(pred)
    p_t = ((labels * prediction_probabilities) +
           ((1 - labels) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if gamma:
        modulating_factor = tf.pow(1.0 - p_t, gamma)
    alpha_weight_factor = 1.0
    if alpha is not None:
        alpha_weight_factor = (labels * alpha +
                               (1 - labels) * (1 - alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)

    # compute the normalizer: the number of positive anchors
    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
    # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

    return tf.reduce_sum(focal_cross_entropy_loss) / normalizer


def _smooth_l1_loss_base(bbox_pred, bbox_targets, sigma=1.0):
    '''

    :param bbox_pred: [-1, 4] in RPN. [-1, cls_num+1, 4] in Fast-rcnn
    :param bbox_targets: shape is same as bbox_pred
    :param sigma:
    :return:
    '''
    sigma_2 = sigma**2

    box_diff = bbox_pred - bbox_targets

    abs_box_diff = tf.abs(box_diff)

    smoothL1_sign = tf.stop_gradient(
        tf.to_float(tf.less(abs_box_diff, 1. / sigma_2)))
    loss_box = tf.pow(box_diff, 2) * (sigma_2 / 2.0) * smoothL1_sign \
               + (abs_box_diff - (0.5 / sigma_2)) * (1.0 - smoothL1_sign)
    return loss_box


def smooth_l1_loss_rcnn(bbox_targets, bbox_pred, anchor_state, sigma=3.0):

    outside_mask = tf.stop_gradient(tf.to_float(tf.greater(anchor_state, 0)))

    bbox_pred = tf.reshape(bbox_pred, [-1, 1, 4])
    bbox_targets = tf.reshape(bbox_targets, [-1, 1, 4])

    value = _smooth_l1_loss_base(bbox_pred,
                                 bbox_targets,
                                 sigma=sigma)
    value = tf.reduce_sum(value, 2)
    value = tf.reshape(value, [-1, 1])

    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    bbox_loss = tf.reduce_sum(
        tf.reduce_sum(value, 1)*outside_mask) / normalizer

    return bbox_loss


def smooth_l1_loss(targets, preds, anchor_state, sigma=3.0):

    sigma_squared = sigma ** 2
    indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
    preds = tf.gather(preds, indices)
    targets = tf.gather(targets, indices)
    regression_diff = preds - targets
    regression_diff = tf.abs(regression_diff)
    regression_loss = tf.where(
        tf.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
    # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

    return tf.reduce_sum(regression_loss) / normalizer


def regress_smooth_l1_loss(targets,preds,anchor_state,anchors,sigma=3.0):

    targets=tf.reshape(targets,[-1,8])

    sigma_squared = sigma ** 2
    indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
    preds = tf.gather(preds, indices)
    targets = tf.gather(targets, indices)
    anchors=tf.gather(anchors,indices)

    #change from delta to abslote data
    if cfgs.METHOD == 'H':
        x_c = (anchors[:, 2] + anchors[:, 0]) / 2
        y_c = (anchors[:, 3] + anchors[:, 1]) / 2
        w= anchors[:, 2] - anchors[:, 0] + 1
        h= anchors[:, 3] - anchors[:, 1] + 1
        # theta = -90 * tf.ones_like(x_c)
        anchors = tf.transpose(tf.stack([x_c, y_c, w, h]))

    preds=bbox_transform.rbbox_transform_inv_1(boxes=anchors, deltas=preds)
    targets=bbox_transform.rbbox_transform_inv_1(boxes=anchors, deltas=targets)


    # prepare for normalization 
    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # loss1
    loss1_1=abs(preds[:,0]-targets[:,0])/anchors[:,2]
    loss1_2=abs(preds[:,1]-targets[:,1])/anchors[:,3]
    loss1_3=abs(preds[:,2]-targets[:,2])/anchors[:,2]
    loss1_4=abs(preds[:,3]-targets[:,3])/anchors[:,3]
    loss1_5=abs(preds[:,4]-targets[:,4])/anchors[:,2]
    loss1_6=abs(preds[:,5]-targets[:,5])/anchors[:,3]
    loss1_7=abs(preds[:,6]-targets[:,6])/anchors[:,2]
    loss1_8=abs(preds[:,7]-targets[:,7])/anchors[:,3]
    box_diff_1=tf.stack([loss1_1,loss1_2,loss1_3,loss1_4,loss1_5,loss1_6,loss1_7,loss1_8],1)
    box_diff_1 = tf.abs(box_diff_1) 
    loss_1 = tf.where(
        tf.less(box_diff_1, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_1, 2),
        box_diff_1 - 0.5 / sigma_squared
    )
    loss_1=tf.reduce_sum(loss_1,1)


    # loss2
    loss2_1=abs(preds[:,0]-targets[:,2])/anchors[:,2]
    loss2_2=abs(preds[:,1]-targets[:,3])/anchors[:,3]
    loss2_3=abs(preds[:,2]-targets[:,4])/anchors[:,2]
    loss2_4=abs(preds[:,3]-targets[:,5])/anchors[:,3]
    loss2_5=abs(preds[:,4]-targets[:,6])/anchors[:,2]
    loss2_6=abs(preds[:,5]-targets[:,7])/anchors[:,3]
    loss2_7=abs(preds[:,6]-targets[:,0])/anchors[:,2]
    loss2_8=abs(preds[:,7]-targets[:,1])/anchors[:,3]
    box_diff_2=tf.stack([loss2_1,loss2_2,loss2_3,loss2_4,loss2_5,loss2_6,loss2_7,loss2_8],1)
    box_diff_2 = tf.abs(box_diff_2) 
    loss_2 = tf.where(
        tf.less(box_diff_2, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_2, 2),
        box_diff_2 - 0.5 / sigma_squared
    )
    loss_2=tf.reduce_sum(loss_2,1)



    # loss3
    loss3_1=abs(preds[:,0]-targets[:,4])/anchors[:,2]
    loss3_2=abs(preds[:,1]-targets[:,5])/anchors[:,3]
    loss3_3=abs(preds[:,2]-targets[:,6])/anchors[:,2]
    loss3_4=abs(preds[:,3]-targets[:,7])/anchors[:,3]
    loss3_5=abs(preds[:,4]-targets[:,0])/anchors[:,2]
    loss3_6=abs(preds[:,5]-targets[:,1])/anchors[:,3]
    loss3_7=abs(preds[:,6]-targets[:,2])/anchors[:,2]
    loss3_8=abs(preds[:,7]-targets[:,3])/anchors[:,3]
    box_diff_3=tf.stack([loss3_1,loss3_2,loss3_3,loss3_4,loss3_5,loss3_6,loss3_7,loss3_8],1)
    box_diff_3 = tf.abs(box_diff_3) 
    loss_3 = tf.where(
        tf.less(box_diff_3, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_3, 2),
        box_diff_3 - 0.5 / sigma_squared
    )
    loss_3=tf.reduce_sum(loss_3,1)    # loss3=tf.reduce_sum(regression_diff_3) / normalizer


    # loss4
    loss4_1=abs(preds[:,0]-targets[:,6])/anchors[:,2]
    loss4_2=abs(preds[:,1]-targets[:,7])/anchors[:,3]
    loss4_3=abs(preds[:,2]-targets[:,0])/anchors[:,2]
    loss4_4=abs(preds[:,3]-targets[:,1])/anchors[:,3]
    loss4_5=abs(preds[:,4]-targets[:,2])/anchors[:,2]
    loss4_6=abs(preds[:,5]-targets[:,3])/anchors[:,3]
    loss4_7=abs(preds[:,6]-targets[:,4])/anchors[:,2]
    loss4_8=abs(preds[:,7]-targets[:,5])/anchors[:,3]
    box_diff_4=tf.stack([loss4_1,loss4_2,loss4_3,loss4_4,loss4_5,loss4_6,loss4_7,loss4_8],1)
    box_diff_4 = tf.abs(box_diff_4) 
    # print(box_diff_4.shape)
    loss_4 = tf.where(
        tf.less(box_diff_4, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_4, 2),
        box_diff_4 - 0.5 / sigma_squared
    )
    loss_4=tf.reduce_sum(loss_4,1)    

    loss=tf.minimum(tf.minimum(loss_1,loss_2),tf.minimum(loss_3,loss_4))
    loss=tf.reduce_sum(loss) / normalizer
    # print(loss.shape)
    # exit()

    return loss



def iou_smooth_l1_loss(targets, preds, anchor_state, target_boxes, anchors, sigma=3.0):
    if cfgs.METHOD == 'H':
        x_c = (anchors[:, 2] + anchors[:, 0]) / 2
        y_c = (anchors[:, 3] + anchors[:, 1]) / 2
        h = anchors[:, 2] - anchors[:, 0] + 1
        w = anchors[:, 3] - anchors[:, 1] + 1
        theta = -90 * tf.ones_like(x_c)
        anchors = tf.transpose(tf.stack([x_c, y_c, w, h, theta]))

    sigma_squared = sigma ** 2
    indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])

    preds = tf.gather(preds, indices)
    targets = tf.gather(targets, indices)
    target_boxes = tf.gather(target_boxes, indices)
    anchors = tf.gather(anchors, indices)

    boxes_pred = bbox_transform.rbbox_transform_inv(boxes=anchors, deltas=preds)

    # compute smooth L1 loss
    # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
    #        |x| - 0.5 / sigma / sigma    otherwise
    regression_diff = preds - targets
    regression_diff = tf.abs(regression_diff)
    regression_loss = tf.where(
        tf.less(regression_diff, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(regression_diff, 2),
        regression_diff - 0.5 / sigma_squared
    )

    overlaps = tf.py_func(iou_rotate_calculate2,
                          inp=[tf.reshape(boxes_pred, [-1, 5]), tf.reshape(target_boxes[:, :-1], [-1, 5])],
                          Tout=[tf.float32])

    overlaps = tf.reshape(overlaps, [-1, 1])
    regression_loss = tf.reshape(tf.reduce_sum(regression_loss, axis=1), [-1, 1])
    iou_factor = tf.stop_gradient(-1 * tf.log(overlaps)) / (tf.stop_gradient(regression_loss) + cfgs.EPSILON)
    # iou_factor = tf.Print(iou_factor, [iou_factor], 'iou_factor', summarize=50)

    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # normalizer = tf.stop_gradient(tf.cast(tf.equal(anchor_state, 1), tf.float32))
    # normalizer = tf.maximum(tf.reduce_sum(normalizer), 1)

    return tf.reduce_sum(regression_loss * iou_factor) / normalizer


def re_order(bboxes):
    n=len(bboxes)
    targets=[]
    for i in range(n):
        box=bboxes[i]
        # 寻找x1
        x1=box[0]
        y1=box[1]
        x1_index=0
        for j in range(1,4):
            if box[2*j]>x1:
                continue
            elif box[2*j]<x1:
                x1=box[2*j]
                y1=box[2*j+1]
                x1_index=j
            else:
                if box[2*j+1]<y1:
                    x1=box[2*j]
                    y1=box[2*j+1]  
                    x1_index=j
                else:
                    continue

        #寻找与x1连线中间点
        for j in range(4):
            if j==x1_index:
                continue
            x_=box[2*j]
            y_=box[2*j+1]
            x_index=j
            val=[]
            for k in range(4):
                if k==x_index or k==x1_index:
                    continue
                else:
                    x=box[2*k]
                    y=box[2*k+1]
                    if x1==x_:
                        val.append(x-x1)
                    else:
                        val1=(y-y1)-(y_-y1)/(x_-x1)*(x-x1)
                        val.append(val1)
            if val[0]*val[1]<0:
                x3=x_
                y3=y_
                for k in range(4):
                    if k==x_index or k==x1_index:
                        continue   
                    x=box[2*k]
                    y=box[2*k+1]
                    if not x1==x_:
                        val=(y-y1)-(y_-y1)/(x_-x1)*(x-x1)    
                        if val>0:
                            x2=x
                            y2=y
                        if val<0:
                            x4=x
                            y4=y    
                    else:
                        val=x1-x
                        if val>0:
                            x2=x
                            y2=y
                        if val<0:
                            x4=x
                            y4=y
                break                        

        targets.append([x1,y1,x2,y2,x3,y3,x4,y4]) 
    return np.array(targets,np.float32)   


def smooth_cl1_loss_4p_align(targets,preds,anchor_state,anchors,sigma=3.0):

    targets=tf.reshape(targets[:, :-1], [-1, 8])

    sigma_squared = sigma ** 2
    indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
    preds = tf.gather(preds, indices)
    targets = tf.gather(targets, indices)
    anchors=tf.gather(anchors,indices)

    #change from delta to abslote data
    if cfgs.METHOD == 'H':
        x_c = (anchors[:, 2] + anchors[:, 0]) / 2
        y_c = (anchors[:, 3] + anchors[:, 1]) / 2
        w= anchors[:, 2] - anchors[:, 0] + 1
        h= anchors[:, 3] - anchors[:, 1] + 1
        # theta = -90 * tf.ones_like(x_c)
        anchors = tf.transpose(tf.stack([x_c, y_c, w, h]))

    preds=bbox_transform.rbbox_transform_inv_1(boxes=anchors, deltas=preds)
    # targets=bbox_transform.rbbox_transform_inv_1(boxes=anchors, deltas=targets)
    targets= tf.py_func(func=re_order,
                        inp=[targets],
                        Tout=[tf.float32]) 
    targets=tf.reshape(targets,[-1,8])
       

    # prepare for normalization 
    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # loss1
    loss1_1=(preds[:,0]-targets[:,0])/anchors[:,2]
    loss1_2=(preds[:,1]-targets[:,1])/anchors[:,3]
    loss1_3=(preds[:,2]-targets[:,2])/anchors[:,2]
    loss1_4=(preds[:,3]-targets[:,3])/anchors[:,3]
    loss1_5=(preds[:,4]-targets[:,4])/anchors[:,2]
    loss1_6=(preds[:,5]-targets[:,5])/anchors[:,3]
    loss1_7=(preds[:,6]-targets[:,6])/anchors[:,2]
    loss1_8=(preds[:,7]-targets[:,7])/anchors[:,3]
    box_diff_1=tf.stack([loss1_1,loss1_2,loss1_3,loss1_4,loss1_5,loss1_6,loss1_7,loss1_8],1)
    box_diff_1 = tf.abs(box_diff_1) 
    loss_1 = tf.where(
        tf.less(box_diff_1, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_1, 2),
        box_diff_1 - 0.5 / sigma_squared
    )
    loss_1=tf.reduce_sum(loss_1,1)

    # loss2
    loss2_1=(preds[:,0]-targets[:,2])/anchors[:,2]
    loss2_2=(preds[:,1]-targets[:,3])/anchors[:,3]
    loss2_3=(preds[:,2]-targets[:,4])/anchors[:,2]
    loss2_4=(preds[:,3]-targets[:,5])/anchors[:,3]
    loss2_5=(preds[:,4]-targets[:,6])/anchors[:,2]
    loss2_6=(preds[:,5]-targets[:,7])/anchors[:,3]
    loss2_7=(preds[:,6]-targets[:,0])/anchors[:,2]
    loss2_8=(preds[:,7]-targets[:,1])/anchors[:,3]
    box_diff_2=tf.stack([loss2_1,loss2_2,loss2_3,loss2_4,loss2_5,loss2_6,loss2_7,loss2_8],1)
    box_diff_2 = tf.abs(box_diff_2) 
    loss_2 = tf.where(
        tf.less(box_diff_2, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_2, 2),
        box_diff_2 - 0.5 / sigma_squared
    )
    loss_2=tf.reduce_sum(loss_2,1)

    # loss1
    loss3_1=(preds[:,0]-targets[:,6])/anchors[:,2]
    loss3_2=(preds[:,1]-targets[:,7])/anchors[:,3]
    loss3_3=(preds[:,2]-targets[:,0])/anchors[:,2]
    loss3_4=(preds[:,3]-targets[:,1])/anchors[:,3]
    loss3_5=(preds[:,4]-targets[:,2])/anchors[:,2]
    loss3_6=(preds[:,5]-targets[:,3])/anchors[:,3]
    loss3_7=(preds[:,6]-targets[:,4])/anchors[:,2]
    loss3_8=(preds[:,7]-targets[:,5])/anchors[:,3]
    box_diff_3=tf.stack([loss3_1,loss3_2,loss3_3,loss3_4,loss3_5,loss3_6,loss3_7,loss3_8],1)
    box_diff_3 = tf.abs(box_diff_3) 
    loss_3 = tf.where(
        tf.less(box_diff_3, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_3, 2),
        box_diff_3 - 0.5 / sigma_squared
    )
    loss_3=tf.reduce_sum(loss_3,1)
    

    loss=tf.minimum(tf.minimum(loss_1,loss_2),loss_3)
    loss=tf.reduce_sum(loss) / normalizer

    return loss

def smooth_cl1_loss_4p_align1(targets,preds,anchor_state,anchors,sigma=3.0):

    targets=tf.reshape(targets, [-1, 8])

    sigma_squared = sigma ** 2
    indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
    preds = tf.gather(preds, indices)
    targets = tf.gather(targets, indices)
    anchors=tf.gather(anchors,indices)

    preds=bbox_transform.rbbox_transform_inv_2(boxes=anchors, deltas=preds)
    targets=bbox_transform.rbbox_transform_inv_2(boxes=anchors, deltas=targets)
    # targets=bbox_transform.rbbox_transform_inv_1(boxes=anchors, deltas=targets)


    #change from delta to abslote data
    if cfgs.METHOD == 'H':
        x_c = (anchors[:, 2] + anchors[:, 0]) / 2
        y_c = (anchors[:, 3] + anchors[:, 1]) / 2
        w= anchors[:, 2] - anchors[:, 0] + 1
        h= anchors[:, 3] - anchors[:, 1] + 1
        # theta = -90 * tf.ones_like(x_c)
        anchors = tf.transpose(tf.stack([x_c, y_c, w, h]))


    targets= tf.py_func(func=re_order,
                        inp=[targets],
                        Tout=[tf.float32]) 
    targets=tf.reshape(targets,[-1,8])
       

    # prepare for normalization 
    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # loss1
    loss1_1=(preds[:,0]-targets[:,0])/anchors[:,2]
    loss1_2=(preds[:,1]-targets[:,1])/anchors[:,3]
    loss1_3=(preds[:,2]-targets[:,2])/anchors[:,2]
    loss1_4=(preds[:,3]-targets[:,3])/anchors[:,3]
    loss1_5=(preds[:,4]-targets[:,4])/anchors[:,2]
    loss1_6=(preds[:,5]-targets[:,5])/anchors[:,3]
    loss1_7=(preds[:,6]-targets[:,6])/anchors[:,2]
    loss1_8=(preds[:,7]-targets[:,7])/anchors[:,3]
    box_diff_1=tf.stack([loss1_1,loss1_2,loss1_3,loss1_4,loss1_5,loss1_6,loss1_7,loss1_8],1)
    box_diff_1 = tf.abs(box_diff_1) 
    loss_1 = tf.where(
        tf.less(box_diff_1, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_1, 2),
        box_diff_1 - 0.5 / sigma_squared
    )
    loss_1=tf.reduce_sum(loss_1,1)

    # loss2
    loss2_1=(preds[:,0]-targets[:,2])/anchors[:,2]
    loss2_2=(preds[:,1]-targets[:,3])/anchors[:,3]
    loss2_3=(preds[:,2]-targets[:,4])/anchors[:,2]
    loss2_4=(preds[:,3]-targets[:,5])/anchors[:,3]
    loss2_5=(preds[:,4]-targets[:,6])/anchors[:,2]
    loss2_6=(preds[:,5]-targets[:,7])/anchors[:,3]
    loss2_7=(preds[:,6]-targets[:,0])/anchors[:,2]
    loss2_8=(preds[:,7]-targets[:,1])/anchors[:,3]
    box_diff_2=tf.stack([loss2_1,loss2_2,loss2_3,loss2_4,loss2_5,loss2_6,loss2_7,loss2_8],1)
    box_diff_2 = tf.abs(box_diff_2) 
    loss_2 = tf.where(
        tf.less(box_diff_2, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_2, 2),
        box_diff_2 - 0.5 / sigma_squared
    )
    loss_2=tf.reduce_sum(loss_2,1)

    # loss1
    loss3_1=(preds[:,0]-targets[:,6])/anchors[:,2]
    loss3_2=(preds[:,1]-targets[:,7])/anchors[:,3]
    loss3_3=(preds[:,2]-targets[:,0])/anchors[:,2]
    loss3_4=(preds[:,3]-targets[:,1])/anchors[:,3]
    loss3_5=(preds[:,4]-targets[:,2])/anchors[:,2]
    loss3_6=(preds[:,5]-targets[:,3])/anchors[:,3]
    loss3_7=(preds[:,6]-targets[:,4])/anchors[:,2]
    loss3_8=(preds[:,7]-targets[:,5])/anchors[:,3]
    box_diff_3=tf.stack([loss3_1,loss3_2,loss3_3,loss3_4,loss3_5,loss3_6,loss3_7,loss3_8],1)
    box_diff_3 = tf.abs(box_diff_3) 
    loss_3 = tf.where(
        tf.less(box_diff_3, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_3, 2),
        box_diff_3 - 0.5 / sigma_squared
    )
    loss_3=tf.reduce_sum(loss_3,1)
    

    loss=tf.minimum(tf.minimum(loss_1,loss_2),loss_3)
    loss=tf.reduce_sum(loss) / normalizer

    return loss

def smooth_continues_modulated_loss_4p(targets,preds,anchor_state,anchors,sigma=3.0):

    targets=tf.reshape(targets[:, :-1], [-1, 8])

    sigma_squared = sigma ** 2
    indices = tf.reshape(tf.where(tf.equal(anchor_state, 1)), [-1, ])
    preds = tf.gather(preds, indices)
    targets = tf.gather(targets, indices)
    anchors=tf.gather(anchors,indices)

    #change from delta to abslote data
    if cfgs.METHOD == 'H':
        x_c = (anchors[:, 2] + anchors[:, 0]) / 2
        y_c = (anchors[:, 3] + anchors[:, 1]) / 2
        w= anchors[:, 2] - anchors[:, 0] + 1
        h= anchors[:, 3] - anchors[:, 1] + 1
        # theta = -90 * tf.ones_like(x_c)
        anchors = tf.transpose(tf.stack([x_c, y_c, w, h]))

    preds=bbox_transform.rbbox_transform_inv_1(boxes=anchors, deltas=preds)
    # targets=bbox_transform.rbbox_transform_inv_1(boxes=anchors, deltas=targets)
    targets= tf.py_func(func=re_order,
                        inp=[targets],
                        Tout=[tf.float32]) 
    targets=tf.reshape(targets,[-1,8])
       

    # prepare for normalization 
    normalizer = tf.stop_gradient(tf.where(tf.equal(anchor_state, 1)))
    normalizer = tf.cast(tf.shape(normalizer)[0], tf.float32)
    normalizer = tf.maximum(1.0, normalizer)

    # loss1
    loss1_1=(preds[:,0]-targets[:,0])/anchors[:,2]
    loss1_2=(preds[:,1]-targets[:,1])/anchors[:,3]
    loss1_3=(preds[:,2]-targets[:,2])/anchors[:,2]
    loss1_4=(preds[:,3]-targets[:,3])/anchors[:,3]
    loss1_5=(preds[:,4]-targets[:,4])/anchors[:,2]
    loss1_6=(preds[:,5]-targets[:,5])/anchors[:,3]
    loss1_7=(preds[:,6]-targets[:,6])/anchors[:,2]
    loss1_8=(preds[:,7]-targets[:,7])/anchors[:,3]
    box_diff_1=tf.stack([loss1_1,loss1_2,loss1_3,loss1_4,loss1_5,loss1_6,loss1_7,loss1_8],1)
    box_diff_1 = tf.abs(box_diff_1) 
    loss_1 = tf.where(
        tf.less(box_diff_1, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_1, 2),
        box_diff_1 - 0.5 / sigma_squared
    )
    loss_1=tf.reduce_sum(loss_1,1)

    # loss2
    loss2_1=(preds[:,0]-targets[:,2])/anchors[:,2]
    loss2_2=(preds[:,1]-targets[:,3])/anchors[:,3]
    loss2_3=(preds[:,2]-targets[:,4])/anchors[:,2]
    loss2_4=(preds[:,3]-targets[:,5])/anchors[:,3]
    loss2_5=(preds[:,4]-targets[:,6])/anchors[:,2]
    loss2_6=(preds[:,5]-targets[:,7])/anchors[:,3]
    loss2_7=(preds[:,6]-targets[:,0])/anchors[:,2]
    loss2_8=(preds[:,7]-targets[:,1])/anchors[:,3]
    box_diff_2=tf.stack([loss2_1,loss2_2,loss2_3,loss2_4,loss2_5,loss2_6,loss2_7,loss2_8],1)
    box_diff_2 = tf.abs(box_diff_2) 
    loss_2 = tf.where(
        tf.less(box_diff_2, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_2, 2),
        box_diff_2 - 0.5 / sigma_squared
    )
    loss_2=tf.reduce_sum(loss_2,1)

    # loss1
    loss3_1=(preds[:,0]-targets[:,6])/anchors[:,2]
    loss3_2=(preds[:,1]-targets[:,7])/anchors[:,3]
    loss3_3=(preds[:,2]-targets[:,0])/anchors[:,2]
    loss3_4=(preds[:,3]-targets[:,1])/anchors[:,3]
    loss3_5=(preds[:,4]-targets[:,2])/anchors[:,2]
    loss3_6=(preds[:,5]-targets[:,3])/anchors[:,3]
    loss3_7=(preds[:,6]-targets[:,4])/anchors[:,2]
    loss3_8=(preds[:,7]-targets[:,5])/anchors[:,3]
    box_diff_3=tf.stack([loss3_1,loss3_2,loss3_3,loss3_4,loss3_5,loss3_6,loss3_7,loss3_8],1)
    box_diff_3 = tf.abs(box_diff_3) 
    loss_3 = tf.where(
        tf.less(box_diff_3, 1.0 / sigma_squared),
        0.5 * sigma_squared * tf.pow(box_diff_3, 2),
        box_diff_3 - 0.5 / sigma_squared
    )
    loss_3=tf.reduce_sum(loss_3,1)
    

    # loss=tf.minimum(tf.minimum(loss_1,loss_2),loss_3)
    loss=-tf.log(tf.exp(-10*loss_1)+tf.exp(-10*loss_2)+tf.exp(-10*loss_3))/10
    loss=tf.reduce_sum(loss) / normalizer

    return loss