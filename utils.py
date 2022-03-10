import functools

import tensorflow as tf
import tensorflow.contrib.slim as slim


# ==============================================================================
# =                                 operations                                 =
# ==============================================================================

def tile_concat(a_list, b_list=[]):
    # tile all elements of `b_list` and then concat `a_list + b_list` along the channel axis
    # `a` shape: (N, H, W, C_a)
    # `b` shape: can be (N, 1, 1, C_b) or (N, C_b)
    a_list = list(a_list) if isinstance(a_list, (list, tuple)) else [a_list]
    b_list = list(b_list) if isinstance(b_list, (list, tuple)) else [b_list]
    print("x shape",len(a_list),",",a_list[0].shape)
    print("att shape:",len(b_list),"特征：",b_list)
    for i, b in enumerate(b_list):
        b = tf.reshape(b, [-1, 1, 1, b.shape[-1]])
        b = tf.tile(b, [1, a_list[0].shape[1], a_list[0].shape[2], 1])#行列复制成图片大小
        b_list[i] = b
    ab_list=tf.concat(a_list + b_list, axis=-1)
    print("拼接后 shape", ab_list.shape)
    return ab_list#按照最后一个纬度拼接起来？？？？

'''
def tile_concat_male(a_list, b_list=[]):
    # tile all elements of `b_list` and then concat `a_list + b_list` along the channel axis
    # `a` shape: (N, H, W, C_a)
    # `b` shape: can be (N, 1, 1, C_b) or (N, C_b)
    a_list = list(a_list) if isinstance(a_list, (list, tuple)) else [a_list]
    b_list = list(b_list) if isinstance(b_list, (list, tuple)) else [b_list]
    print("x shape",a_list)

    for i, b in enumerate(b_list):
        print("att shape" ,b.shape)
        b = tf.reshape(b, [-1, 1, 1, b.shape[-1]])
        b = tf.tile(b, [1, a_list[0].shape[1], a_list[0].shape[2],  1])#行列复制成图片大小
        b_list[i] = b
    return tf.concat(a_list + b_list, axis=-1)#按照最后一个纬度拼接起来？？？？
'''

# ==============================================================================
# =                                   others                                   =
# ==============================================================================

def get_norm_layer(norm, training, updates_collections=None):
    if norm == 'none':
        return lambda x: x
    elif norm == 'batch_norm':
        return functools.partial(slim.batch_norm, scale=True, is_training=training, updates_collections=updates_collections)
    elif norm == 'instance_norm':
        return slim.instance_norm
    elif norm == 'layer_norm':
        return slim.layer_norm
