import tensorflow as tf
import tensorflow_addons as tfa
# from tensorflow.keras.backend import epsilon
# import sys
# tf.compat.v1.disable_eager_execution()
# @tf.function

def compute_image_loss(inputs, weights=[1., 1.], kernel_size=1, kernel_depth=256, **kwargs):
    """Compute the loss for one image
    inputs:
    cls_targets: class targets (flattened over FPN and spatial dims, one-hot) -> shape [locations, ncls]
    labels_targets: labels of positive locations ((flattened over FPN and spatial dims) [locations]
    gt_mask: gt masks with labeled instances (H, W)
    cls_pred: predicted class logits (flattened over FPN and spatial dims -> shape [locations, ncls])
    kernel_pred: predicted kernel maps (flattened over FPN and spatial dims -> shape [locations, ks*ks*mask_head_depth])
    mask_head_pred: output of the mask head [H, W, mask_head_depth]
    To compute mask loss we take predicted kernels at gt positive locations

    """

    # bboxes loss (one box per location inside objects only)
    # takes locations where the gt class is not bg

    cls_targets, labels_targets, gt_masks, cls_pred, kernel_pred, mask_head_pred = inputs
    labels, _, counts = tf.unique_with_counts(labels_targets)
    # tf.print(labels)
    has_objects = tf.math.reduce_sum(gt_masks,axis=(-1,-2)) >1
    max_label = tf.reduce_max(labels)
    pos_idx = tf.where(labels_targets > 0)
    # tf.print("f",has_objects)
    
    fx = None
    def check_mask_size(label,kernel_preds,pos_idx):
        def seg_loss():
            
            return 0.0
            # gt_mask_img = tf.expand_dims(gt_mask_img,axis=0)
            # gt_boxes = tf.expand_dims(gt_boxes,axis=0)
            # gt_cls_ids = tf.expand_dims(gt_cls_ids,axis=0)
            # gt_labels = tf.expand_dims(gt_labels,axis=0)
        def cal_seg_loss():
            ohe_masks = tf.one_hot(gt_masks, max_label+1)[..., 1:] # max label + 1 for bg
            mask_targets = tf.TensorArray(tf.float32, size=0, dynamic_size=True, element_shape=ohe_masks.shape[:-1])
            # x
            # tf.print("x",ohe_masks)
            # x
            for i in tf.range(tf.shape(pos_idx)[0]):

                mask_targets = mask_targets.write(tf.cast(i, tf.int32), ohe_masks[..., labels_targets[pos_idx[i,0]] - 1]) #labels - 1 because label 1 is at slice 0
                # tf.print("hey",ohe_masks[..., labels_targets[pos_idx[i,0]] - 1])
                # mask_targets = tf.concat([mask_targets, ohe_masks[..., labels_targets[i], tf.newaxis]], axis=-1)
            # tf.print("hwew",mask_targets)
            mask_targets = mask_targets.stack()
            # tf.print(masks_)
            # mask_targets = tf.transpose(mask_targets, [1,2,0]) pas besoin de transposer ici, la prédiction a le même format [ninst, H, W]
            # mask_targets = tf.repeat(ohe_masks, repeats=labels, axis=-1)

            # Get the kernels corresponding to positive GT
            kernel_pred = tf.gather(kernel_preds, pos_idx[:, 0])  # shape [n_inst, kernel_depth*kernel_size**2]
            kernel_pred = tf.transpose(kernel_pred)
            kernel_pred = tf.reshape(kernel_pred, (kernel_size, kernel_size, kernel_depth, -1)) # SHAPE [ks, ks, cin, cout]
            seg_preds = tf.sigmoid(tf.nn.conv2d(mask_head_pred[tf.newaxis, ...], kernel_pred, strides=1, padding="SAME")) # shape [1, H, W, ninstances]
            seg_preds = tf.transpose(seg_preds[0], perm=[2, 0, 1])  # reshape to [ninstances, H, W]
            # tf.print("xy",seg_preds)
            seg_loss = dice_loss(seg_preds, mask_targets)
            return seg_loss
        return tf.cond(tf.size(label)==3,seg_loss,cal_seg_loss)
    # tf.print(len(labels))
    seg_loss = check_mask_size(labels,kernel_pred,pos_idx)
    # tf.print("test", tf.one_hot(gt_masks[has_objects], max_label+1))
    
    # tf.print("label",max_label,tf.raw_ops.UniqueWithCountsV2(x=gt_masks,axis=[1]))
    # ohe_masks = tf.reshape(ohe_masks,[-1])
    # print("OHE GT MASK", ohe_masks.shape, ohe_masks.shape[:-1])
    # Here, each gt pixel is associated with a mask. Pixels of the same object are associated with the same masks
        # tf.print("gt_masks",gt_masks)
        # tf.print("x1",ohe_masks)
        
    cls_loss = focal_loss(cls_pred, cls_targets)
    

    return cls_loss * weights[0], seg_loss * weights[1]


def focal_loss(pred, gt, alpha=0.25, gamma=2.0):
    pred, gt = tf.reshape(pred, (-1, 1)), tf.reshape(gt, (-1, 1))
    anchor_obj_count = tf.cast(tf.math.count_nonzero(gt), pred.dtype)
    alpha_factor = tf.ones_like(gt) * alpha
    alpha_factor = tf.where(tf.equal(gt, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(tf.equal(gt, 1), 1 - pred, pred)
    focal_weight = alpha_factor * focal_weight**gamma / (anchor_obj_count + 1)
    return tf.losses.BinaryCrossentropy(reduction='sum', from_logits=False)(gt, pred, sample_weight=focal_weight)


def dice_loss(pred, gt):
    a = tf.reduce_sum(pred * gt)
    b = tf.reduce_sum(pred * pred)
    c = tf.reduce_sum(gt * gt)
    dice = tf.math.divide_no_nan((2 * a), (b + c))
    return 1 - dice  # tf.where(dice > 0, dice, 1.)
