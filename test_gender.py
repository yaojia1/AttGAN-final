import os

import imlib as im
import numpy as np
import pylib as py
import tensorflow as tf
import tflib as tl
import tqdm

import data
import module_gender as module


# ==============================================================================
# =                                   param                                    =
# ==============================================================================

py.arg('--img_dir', default='./data/CelebAMask-HQ/CelebA-HQ-img')#'./data/img_celeba/aligned/align_size(572,572)_move(0.250,0.000)_face_factor(0.450)_jpg/data')
py.arg('--test_label_path', default='./data/CelebAMask-HQ/test_label.txt')
py.arg('--test_att_names', choices=data.ATT_ID.keys(), nargs='+', default=['Male'])#default=['Bangs', 'Mustache'])
py.arg('--test_ints', type=float, nargs='+', default=2)

py.arg('--experiment_name', default='default')

py.arg('--test_lendata', default=64)
py.arg('--test_data_skip', default=500)
args_ = py.args()

# output_dir
output_dir = py.join('output', args_.experiment_name)

# save settings
args = py.args_from_yaml(py.join(output_dir, 'settings_male.yml'))
args.__dict__.update(args_.__dict__)

# others
n_atts = len(args.att_names)
if not isinstance(args.test_ints, list):
    args.test_ints = [args.test_ints] * len(args.test_att_names)
elif len(args.test_ints) == 1:
    args.test_ints = args.test_ints * len(args.test_att_names)

sess = tl.session()
sess.__enter__()  # make default


# ==============================================================================
# =                               data and model                               =
# ==============================================================================

# data
#TODO 修改数据库大小
test_dataset, len_test_dataset = data.make_celeba_dataset(args.img_dir, args.test_label_path, args.att_names, args.n_samples,
                                                          load_size=args.load_size, crop_size=args.crop_size,
                                                          training=False, drop_remainder=False, shuffle=False, repeat=None,
                                                          test_begin=args.test_data_skip,
                                                          test_num=args.test_lendata)
test_iter = test_dataset.make_one_shot_iterator()


# ==============================================================================
# =                                   graph                                    =
# ==============================================================================

def sample_graph():
    # ======================================
    # =               graph                =
    # ======================================

    test_next = test_iter.get_next()

    if not os.path.exists(py.join(output_dir, 'generator.pb')):
        # model
        Genc, Gdec, _ = module.get_model(args.model, n_atts, weight_decay=args.weight_decay)

        # placeholders & inputs
        xa = tf.placeholder(tf.float32, shape=[None, args.crop_size, args.crop_size, 3])
        b_ = tf.placeholder(tf.float32, shape=[None, n_atts])

        # sample graph
        code=Genc(xa, training=False)
        # sample graph
        x,zs = Gdec(code, b_, training=False)
    else:
        # load freezed model
        with tf.gfile.GFile(py.join(output_dir, 'generator.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='generator')

        # placeholders & inputs
        xa = sess.graph.get_tensor_by_name('generator/xa:0')
        b_ = sess.graph.get_tensor_by_name('generator/b_:0')

        # sample graph
        x,zs = sess.graph.get_tensor_by_name('generator/xb:0')

    # ======================================
    # =            run function            =
    # ======================================

    save_dir = './output/%s/samples_testing_gender' % args.experiment_name
    tmp = ''
    for test_att_name, test_int in zip(args.test_att_names, args.test_ints):
        tmp += '_%s_%s' % (test_att_name, '{:g}'.format(test_int))
    save_dir = py.join(save_dir, tmp[1:])
    py.mkdir(save_dir)

    def run():
        cnt = 0
        for _ in tqdm.trange(len_test_dataset):
            # data for sampling
            xa_ipt, a_ipt = sess.run(test_next)
            b_ipt = np.copy(a_ipt)
            for test_att_name in args.test_att_names:
                i = args.att_names.index(test_att_name)
                print(b_ipt.shape,b_ipt[..., i].shape)
                print("女性个数",np.sum(b_ipt[..., i].flatten()==0),"男性个数",np.sum(b_ipt[..., i].flatten()==1))
                #print("女性个数",b_ipt[..., i].count(0),"男性个数",b_ipt[..., i].count(1))
                b_ipt[..., i] = 1 - b_ipt[..., i]#不同纬度维度上选数据，该属性的一个数0或1

                b_ipt = data.check_attribute_conflict(b_ipt, test_att_name, args.att_names)

            b__ipt = (b_ipt * 2 - 1).astype(np.float32)  # !!!
            for test_att_name, test_int in zip(args.test_att_names, args.test_ints):
                i = args.att_names.index(test_att_name)
                b__ipt[..., i] = b__ipt[..., i] * test_int

            x_opt_list = [xa_ipt]
            print("input batch:", len(xa_ipt), "张：", np.array(xa_ipt[0]).shape)
            #编辑性别
            code_opt,x_opt,zs_opt = sess.run([code, x, zs], feed_dict={xa: xa_ipt, b_: b__ipt})
            #重构图片
            rec_opt= sess.run(x, feed_dict={xa: xa_ipt, b_: a_ipt})
            #编码器
            #code_opt=sess.run(code, feed_dict={xa: xa_ipt, b_: b__ipt})
            print("特征编码", len(code_opt),"层：")
            for ini in range(len(code_opt)):
                print("层",ini+1,":",np.array(code_opt[ini]).shape)
            #生成器

            #zs_opt = sess.run(zs, feed_dict={xa: xa_ipt, b_: b__ipt})
            print("生成器", len(x_opt), "张：",np.array(x_opt[0]).shape)
            print("生成器层", len(zs_opt), "层：")
            ii=0
            for ini in range(len(zs_opt)):
                print("层", ini + 1 - ii, ":", np.array(zs_opt[ini]).shape)
                if ii<4:
                    ii+=1


            x_opt_list.append(rec_opt)
            x_opt_list.append(x_opt)
            print("输出新旧图片",np.array(x_opt_list).shape)
            sample = np.transpose(x_opt_list, (1, 2, 0, 3, 4))
            print("转换", sample.shape)
            sample = np.reshape(sample, (sample.shape[0], -1, sample.shape[2] * sample.shape[3], sample.shape[4]))
            print("输出新旧拼接图片", sample.shape)

            for s in sample:
                cnt += 1
                im.imwrite(s, '%s/%d.jpg' % (save_dir, cnt))

    return run


sample = sample_graph()


# ==============================================================================
# =                                    test                                    =
# ==============================================================================

# checkpoint
if not os.path.exists(py.join(output_dir, 'generator.pb')):
    checkpoint = tl.Checkpoint(
        {v.name: v for v in tf.global_variables()},
        py.join(output_dir, 'checkpoints'),
        max_to_keep=1
    )
    checkpoint.restore().run_restore_ops()

sample()
# 网络结构写入
summary_writer = tf.summary.FileWriter('./log/', sess.graph)
sess.close()
