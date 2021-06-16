#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /media/allenyl/DATA/Projects/server_setup/repo-ai/NLP-AI-in-Law/samples/NN_models.py
# Project: /media/allenyl/DATA/Projects/server_setup/repo-ai/NLP-AI-in-Law/samples
# Created Date: Wednesday, January 30th 2019, 8:25:20 pm
# Author: Allenyl(allen7575@gmail.com>)
# -----
# Last Modified: Monday, February 10th 2020, 8:50:53 pm
# Modified By: Allenyl
# -----
# Copyright 2018 - 2019 Allenyl Copyright, Allenyl Company
# -----
# license:
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
# ------------------------------------
# HISTORY:
# Date      	By	Comments
# ----------	---	---------------------------------------------------------
###
# import tensorflow as tf
import tensorflow.compat.v1 as tf
# import tensorflow_addons as tfa
# import keras
from sklearn.utils import shuffle
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import random
import sys


def weight_variable(shape, name='W', seed=None, dtype=tf.float32):
    if seed is not None:
        w = tf.Variable(tf.truncated_normal(
            shape=shape, stddev=0.1, seed=seed), name=name, dtype=dtype)
    else:
        w = tf.Variable(tf.truncated_normal(
            shape=shape, stddev=0.1), name=name, dtype=dtype)
    # summary op
    tf.summary.histogram('weights', w)
    return w


def bias_variable(shape, name='B', dtype=tf.float32):
    b = tf.Variable(tf.constant(0.1, shape=shape), name=name, dtype=dtype)
    # summary op
    tf.summary.histogram('biases', b)
    return b


class DNN():

    def __init__(self, x, y, class_weights=None, class_threshold=None, seed=1234):
        if seed is not None:
            self.seed = seed
            tf.set_random_seed(seed)
        else:
            self.seed = None

        # set variable
        self.x = x
        self.y = y
        self.class_weights = class_weights
        self.class_threshold = class_threshold

        # reset graph
        tf.reset_default_graph()

        # build graph
        self.init_input_place()
        self.h, self.w = self.build_graph()

        # initial op
        self.loss_op()
        self.predict_op()
        self.metrics_op()
        self.train_op()
        init_op = tf.global_variables_initializer()  # define initialize operation

        # initial training curve list
        self.train_loss_list, self.train_acc_list = [], []
        self.valid_loss_list, self.valid_acc_list = [], []

        # initial session
        self.sess = tf.InteractiveSession()  # open new session
        self.sess.run(init_op)  # run initialze operation
        self.prev_loss = float("inf")

    def init_input_place(self):
        with tf.name_scope('input_layer'):
            self.x_in = tf.placeholder(
                shape=[None, self.x.shape[1]], dtype=tf.float32)
            print('input shape:', self.x_in.shape)

            self.y_out = tf.placeholder(
                shape=[None, self.y.shape[1]], dtype=tf.float32)
            print('output shape:', self.y_out.shape)

        with tf.name_scope('control_variable'):
            self.keep_prob = tf.placeholder(
                tf.float32, name='dropout_keep_prob')
            print("dropout : ", self.keep_prob)

    def load(self, model_path, meta_path=None):
        if meta_path:
            # load graph
            # note: not yet functioning
            saver = tf.train.import_meta_graph(meta_path)
        else:
            saver = tf.train.Saver()
        saver.restore(self.sess, model_path)
        # show global variables
        gvars = self.sess.graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        print(gvars)

    def save(self, model_path):
        saver = tf.train.Saver()
        saver.save(self.sess, model_path)

    def build_graph(self):
        hidden_layer_list = []
        weight_list = []
        drop_rate = 0.4
        with tf.variable_scope('hidden_layer1'):
            output_shape = int(self.x_in.shape.as_list()[1]*0.5)
            print('shape: (%d, %d)' %
                  (self.x_in.shape.as_list()[1], output_shape))

            # w = tf.get_variable(name='w1', shape=[self.x_in.shape.as_list()[1], output_shape], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            # b = tf.get_variable(name='b1', shape=[output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            w = weight_variable(shape=[self.x_in.shape.as_list()[
                                1], output_shape], name='w1', seed=self.seed)
            b = bias_variable(shape=[output_shape], name='b1')
            out = tf.nn.leaky_relu(tf.add(tf.matmul(self.x_in, w), b))
            hidden_layer_list.append(out)
            weight_list.append(w)

        with tf.variable_scope('hidden_layer2'):
            output_shape = int(out.shape.as_list()[1]*0.5)
            print('shape: (%d, %d)' % (out.shape.as_list()[1], output_shape))

            # w = tf.get_variable(name='w2', shape=[out.shape.as_list()[1], output_shape], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            # b = tf.get_variable(name='b2', shape=[output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            w = weight_variable(
                shape=[out.shape.as_list()[1], output_shape], name='w2', seed=self.seed)
            b = bias_variable(shape=[output_shape], name='b2')
            out = tf.nn.leaky_relu(tf.add(tf.matmul(out, w), b))
            hidden_layer_list.append(out)
            weight_list.append(w)

        with tf.variable_scope('hidden_layer3'):
            output_shape = int(out.shape.as_list()[1]*0.5)
            print('shape: (%d, %d)' % (out.shape.as_list()[1], output_shape))

            # w = tf.get_variable(name='w3', shape=[out.shape.as_list()[1], output_shape], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            # b = tf.get_variable(name='b3', shape=[output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            w = weight_variable(
                shape=[out.shape.as_list()[1], output_shape], name='w3', seed=self.seed)
            b = bias_variable(shape=[output_shape], name='b3')
            out = tf.nn.leaky_relu(tf.add(tf.matmul(out, w), b))
            hidden_layer_list.append(out)
            weight_list.append(w)

        with tf.variable_scope('hidden_layer4'):
            output_shape = int(out.shape.as_list()[1]*0.5)
            print('shape: (%d, %d)' % (out.shape.as_list()[1], output_shape))

            # w = tf.get_variable(name='w4', shape=[out.shape.as_list()[1], output_shape], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            # b = tf.get_variable(name='b4', shape=[output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            w = weight_variable(
                shape=[out.shape.as_list()[1], output_shape], name='w4', seed=self.seed)
            b = bias_variable(shape=[output_shape], name='b4')
            out = tf.nn.leaky_relu(tf.add(tf.matmul(out, w), b))
            hidden_layer_list.append(out)
            weight_list.append(w)

        # with tf.variable_scope('hidden_layer5'):
        #     output_shape = int(out.shape.as_list()[1]*0.5)
        #     print('shape: (%d, %d)' % (out.shape.as_list()[1], output_shape))

        #     # w = tf.get_variable(name='w4', shape=[out.shape.as_list()[1], output_shape], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        #     # b = tf.get_variable(name='b4', shape=[output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
        #     w = weight_variable(
        #         shape=[out.shape.as_list()[1], output_shape], name='w5', seed=self.seed)
        #     b = bias_variable(shape=[output_shape], name='b5')
        #     out = tf.nn.relu(tf.add(tf.matmul(out, w), b))
        #     # out = tf.nn.dropout(out, drop_rate)
        #     hidden_layer_list.append(out)
        #     weight_list.append(w)

        # with tf.variable_scope('hidden_layer5'):
        #     w = tf.get_variable(name='w5', shape=[out.shape[1], 50], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
        #     b = tf.get_variable(name='b5', shape=[w.shape[1]], dtype=tf.float32, initializer=tf.constant_initializer(0.0))
        #     w = weight_variable(shape=[out.shape.as_list()[1], output_shape], name='w5', seed=self.seed)
        #     b = bias_variable(shape=[output_shape], name='b5')
        #     out = tf.nn.relu(tf.add(tf.matmul(out, w),b))

        #     hidden_layer_list.append(out)
        #     weight_list.append(w)

        with tf.variable_scope('output_layer'):
            output_shape = self.y_out.shape.as_list()[1]
            print('shape: (%d, %d)' % (out.shape.as_list()[1], output_shape))

            # w = tf.get_variable(name='wo', shape=[out.shape.as_list()[1], output_shape], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1))
            # b = tf.get_variable(name='bo', shape=[output_shape], dtype=tf.float32, initializer=tf.constant_initializer(0.1))
            w = weight_variable(
                shape=[out.shape.as_list()[1], output_shape], name='wo', seed=self.seed)
            b = bias_variable(shape=[output_shape], name='bo')
            out = tf.add(tf.matmul(out, w), b)
            hidden_layer_list.append(out)
            weight_list.append(w)

        return hidden_layer_list, weight_list

    def loss_op(self):
        with tf.name_scope('cost'):
            self.output_logits = self.h[-1]
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.output_logits, labels=self.y_out)

            if self.class_weights is not None:
                class_weights = tf.constant(self.class_weights)
            else:
                class_weights = tf.constant([1.0])

            weights = tf.reduce_sum(class_weights * self.y_out, axis=1)
            weighted_losses = loss * weights
            # self.print_op = tf.print("weights:", weights)
            self.cost = tf.reduce_mean(weighted_losses)
            # self.cost = tf.reduce_mean(loss)

    def print_cost(self, feed_dict):
        feed_dict2 = {}
        feed_dict2[self.x_in] = feed_dict['x']
        feed_dict2[self.y_out] = feed_dict['y']
        feed_dict2[self.keep_prob] = feed_dict['k']

        result = self.sess.run([self.cost, self.output_label_index,
                                self.output_multi_label_class], feed_dict=feed_dict2)
        # print(result.shape)

        print("\nprint result...")
        for value in result:
            print("shape: ", value.shape)
            print("value: ", value)

    def predict_op(self):
        with tf.name_scope('output_proba'):
            self.output_proba = tf.nn.softmax(self.output_logits)

        with tf.name_scope('output_label_index'):
            self.output_label_index = tf.argmax(self.output_proba, axis=1)

        with tf.name_scope('output_multi_label_class'):
            # suppose r is 3-classes multi-label probability
            r = self.output_proba
            # print(r.shape)

            # set 3 different threshold for each class [0.5, 0.6, 0.3]
            if self.class_threshold is not None:
                cond = tf.greater(r, tf.constant(self.class_threshold))
                # print(cond.shape)
            else:
                cond = tf.greater(r, tf.constant([0.5]))

            # return 0 for condition is true (greater then threshold) or 1 for false (less equal then threshold)
            result = tf.where(cond, tf.fill(tf.shape(r), 1),
                              tf.fill(tf.shape(r), 0))
            # print(result.shape)

            self.output_multi_label_class = result

    def metrics_op(self):
        with tf.name_scope('auc'):
            self.auc = tf.metrics.auc(
                labels=self.y_out, predictions=self.output_proba)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(
                self.output_logits), axis=1), tf.argmax(self.y_out, axis=1))
            self.compute_acc = tf.reduce_mean(
                tf.cast(correct_prediction, dtype=tf.float32))

    def train_op(self):
        with tf.name_scope('train_step'):
            # self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.cost)
            # self.train_step = tfa.optimizers.RectifiedAdam(lr=1e-3).minimize(self.cost)
            from keras_radam.training import RAdamOptimizer
            from lib.lookahead import LookaheadOptimizer

            # opt = tf.train.AdamOptimizer(0.001)
            opt = RAdamOptimizer(learning_rate=1e-3)
            opt = LookaheadOptimizer(opt)
            # self.train_step = RAdamOptimizer(learning_rate=1e-3).minimize(self.cost)
            self.train_step = opt.minimize(self.cost)

    def train(self, x, y, x_valid=None, y_valid=None, keep_prob=1, batchsize=16, epoch=100):

        if x_valid is None or y_valid is None:
            self.noValid = True
        else:
            self.noValid = False

        if self.seed is not None:
            rng = random.Random(self.seed)
        else:
            rng = random.Random()

        for i in tqdm(range(epoch)):
            total_batch = int(np.floor(len(x)/batchsize))
            train_loss_collector, train_acc_collector = [], []

            for j in np.arange(total_batch):
                batch_start = j*batchsize
                batch_stop = (j+1)*batchsize

                x_batch = x[batch_start:batch_stop]
                y_batch = y[batch_start:batch_stop]

                feed_dict = {}
                feed_dict[self.x_in] = x_batch
                feed_dict[self.y_out] = y_batch
                feed_dict[self.keep_prob] = keep_prob

                # print(x_batch.shape)
                # print(y_batch.shape)

                try:
                    self.print_op
                except:
                    # print("no self.print_op")
                    this_loss, this_acc, _ = self.sess.run(
                        [self.cost, self.compute_acc, self.train_step], feed_dict=feed_dict)
                else:
                    # print("use self.print_op")
                    this_loss, this_acc, _, _ = self.sess.run(
                        [self.cost, self.compute_acc, self.train_step, self.print_op], feed_dict=feed_dict)

                # if self.print_op:
                #     # print("use self.print_op")
                #     this_loss, this_acc, _, _ = self.sess.run([self.cost, self.compute_acc, self.train_step, self.print_op], feed_dict=feed_dict)
                # else:
                #     # print("no self.print_op")
                #     this_loss, this_acc, _ = self.sess.run([self.cost, self.compute_acc, self.train_step], feed_dict=feed_dict)

                train_loss_collector.append(this_loss)
                train_acc_collector.append(this_acc)
              

            self.train_loss_list.append(np.mean(train_loss_collector))
            self.train_acc_list.append(np.mean(train_acc_collector))

            if not self.noValid:
                feed_dict = {}
                feed_dict[self.x_in] = x_valid
                feed_dict[self.y_out] = y_valid
                feed_dict[self.keep_prob] = keep_prob

                valid_loss, valid_acc = self.sess.run(
                    [self.cost, self.compute_acc], feed_dict=feed_dict)

                self.valid_loss_list.append(valid_loss)
                self.valid_acc_list.append(valid_acc)
                
                if valid_loss < self.prev_loss:
                    self.prev_loss = valid_loss
                    self.save('./myModel.ckpt')

            seed = rng.randrange(2**32 - 1)
            # print("seed:", seed)
            # x, y = shuffle(x, y, random_state=seed)

        if not self.noValid:
            valid_accuracy = self.get_accuracy(x_valid, y_valid)
            self.get_auc(x_valid, y_valid)
        else:
            valid_accuracy = None

        self.plot()

        return valid_accuracy

    def predict_proba(self, x):
        feed_dict = {}
        feed_dict[self.x_in] = x
        feed_dict[self.keep_prob] = 1

        return self.sess.run([self.output_proba], feed_dict=feed_dict)[0]

    def predict(self, x):
        feed_dict = {}
        feed_dict[self.x_in] = x
        feed_dict[self.keep_prob] = 1

        label_index = self.sess.run(
            [self.output_label_index], feed_dict=feed_dict)[0]
        # print(label_index)

        label = np.zeros(shape=(len(label_index), self.y.shape[1]))
        label[np.arange(len(label_index)), label_index] = 1

        return label

    def predict_multi_label(self, x):
        feed_dict = {}
        feed_dict[self.x_in] = x
        feed_dict[self.keep_prob] = 1

        multi_label_class = self.sess.run(
            [self.output_multi_label_class], feed_dict=feed_dict)[0]
        # print(label_multiclass)

        # label = np.zeros(shape=(len(label_index), self.y.shape[1]))
        # label[np.arange(len(label_index)),label_index]=1

        return multi_label_class

    def transform(self, x):
        feed_dict = {}
        feed_dict[self.x_in] = x
        feed_dict[self.keep_prob] = 1

        return self.sess.run([self.output_logits], feed_dict=feed_dict)[0]

    def get_auc(self, x_test, y_test):
        init = tf.local_variables_initializer()
        self.sess.run(init)

        feed_dict = {}
        feed_dict[self.x_in] = x_test
        feed_dict[self.y_out] = y_test
        feed_dict[self.keep_prob] = 1

        auc, update_op = self.sess.run([self.auc], feed_dict=feed_dict)[0]

        print('auc: ', update_op)
        return update_op

    def get_accuracy(self, x_test, y_test):

        feed_dict = {}
        feed_dict[self.x_in] = x_test
        feed_dict[self.y_out] = y_test
        feed_dict[self.keep_prob] = 1

        test_loss, test_acc = self.sess.run(
            [self.cost, self.compute_acc], feed_dict=feed_dict)

        print('accuracy: ', test_acc)

        return test_acc

    def plot(self):
        plt.title('Loss')
        plt.plot(np.arange(len(self.train_loss_list)),
                 self.train_loss_list, 'b', label='train')
        if not self.noValid:
            plt.plot(np.arange(len(self.valid_loss_list)),
                     self.valid_loss_list, 'r', label='valid')
        plt.legend()
        plt.show()

        plt.title('Accuracy')
        plt.plot(np.arange(len(self.train_acc_list)),
                 self.train_acc_list, 'b', label='train')
        if not self.noValid:
            plt.plot(np.arange(len(self.valid_acc_list)),
                     self.valid_acc_list, 'r', label='valid')
        plt.legend()
        plt.show()


class splited_DNN(DNN):
    def __init__(self, x, y, num_or_size_splits, bottleneck_size=100, class_weights=None, class_threshold=None, seed=1234):
        if seed is not None:
            self.seed = seed
#             tf.set_random_seed(seed)
        else:
            self.seed = None

        # set variable
        self.x = x
        self.y = y
        self.class_weights = class_weights
        self.class_threshold = class_threshold
        self.num_or_size_splits = num_or_size_splits
        self.bottleneck_size = bottleneck_size
        # self.keep_prob = dropout_keep_prob
        self.dropout_rate = 1.0

        # reset graph
        tf.reset_default_graph()

        # build graph
        self.init_input_place()
        self.h, self.w = self.build_graph()

        # initial op
        self.loss_op()
        self.predict_op()
        self.metrics_op()
        self.train_op()
        init_op = tf.global_variables_initializer()  # define initialize operation

        # initial training curve list
        self.train_loss_list, self.train_acc_list = [], []
        self.valid_loss_list, self.valid_acc_list = [], []

        # initial session
        self.sess = tf.InteractiveSession()  # open new session
        self.sess.run(init_op)  # run initialze operation
        self.prev_loss = float("inf")

    def init_input_place(self):
        with tf.name_scope('input_layer'):
            self.x_in = tf.placeholder(
                shape=[None, self.x[0].shape[0]], dtype=tf.float32)
            print('input shape:', self.x_in.shape)

            self.y_out = tf.placeholder(
                shape=[None, self.y[0].shape[0]], dtype=tf.float32)
            print('output shape:', self.y_out.shape)

        with tf.name_scope('control_variable'):
            self.keep_prob = tf.placeholder(
                tf.float32, name='dropout_keep_prob')
            # self.keep_prob = tf.constant([0.6])
            print('dropout-splitedDNN:', self.keep_prob)

    def build_graph(self):
        hidden_layer_list = []
        weight_list = []

        with tf.name_scope('hidden_layer1'):
            tmp_input_list = []
            tmp_weight_list = []
            output_shape = self.bottleneck_size
            print(output_shape, 'output shape')
            print(self.x_in.shape)
            print(self.num_or_size_splits)

            splited_x_in_list = tf.split(
                self.x_in, self.num_or_size_splits, axis=1)

            # print(output_shape)
            # print(splited_x_in_list)

            for splited_x_in in splited_x_in_list:
                print('shape: (%d, %d)' %
                      (splited_x_in.shape.as_list()[1], output_shape))

                # print(splited_x_in.shape.as_list()[1])
                w = weight_variable([splited_x_in.shape.as_list()[
                                    1], output_shape], seed=self.seed)
                b = bias_variable([output_shape])
                h = tf.nn.relu(tf.add(tf.matmul(splited_x_in, w), b))
                out = h
                # out = tf.nn.dropout(out, self.keep_prob)
                out = tf.nn.dropout(out, self.dropout_rate)
                # add to list
                tmp_input_list.append(out)
                tmp_weight_list.append(w)

            hidden_layer_list.append(tmp_input_list)
            weight_list.append(tmp_weight_list)

        with tf.name_scope('hidden_layer2'):
            # print(hidden_layer_list[0])
            out = tf.concat(hidden_layer_list[0], 1)
            output_shape = int(out.shape.as_list()[1]*0.5)
            
            print('shape: (%d, %d)' % (out.shape.as_list()[1], output_shape))

            w = weight_variable(
                [out.shape.as_list()[1], output_shape], seed=self.seed)
            b = bias_variable([output_shape])
            h = tf.nn.relu(tf.add(tf.matmul(out, w), b))
            out = h
            # out = tf.nn.dropout(out, self.keep_prob)
            out = tf.nn.dropout(out, self.dropout_rate)

            hidden_layer_list.append(out)
            weight_list.append(w)

        with tf.name_scope('hidden_layer3'):
            output_shape = int(out.shape.as_list()[1]*0.5)
            print('shape: (%d, %d)' % (out.shape.as_list()[1], output_shape))
            w = weight_variable(
                [out.shape.as_list()[1], output_shape], seed=self.seed)
            b = bias_variable([output_shape])
            h = tf.nn.relu(tf.add(tf.matmul(out, w), b))
            out = h
            # out = tf.nn.dropout(out, self.keep_prob)
            out = tf.nn.dropout(out, self.dropout_rate)

            hidden_layer_list.append(out)
            weight_list.append(w)

        with tf.name_scope('hidden_layer4'):
            output_shape = int(out.shape.as_list()[1]*0.5)
            print('shape: (%d, %d)' % (out.shape.as_list()[1], output_shape))

            w = weight_variable(
                [out.shape.as_list()[1], output_shape], seed=self.seed)
            b = bias_variable([output_shape])
            h = tf.nn.relu(tf.add(tf.matmul(out, w), b))
            out = h
            # out = tf.nn.dropout(out, self.keep_prob)
            out = tf.nn.dropout(out, self.dropout_rate)

            hidden_layer_list.append(out)
            weight_list.append(w)

        with tf.name_scope('hidden_layer5'):
            output_shape = int(out.shape.as_list()[1]*0.5)
            print('shape: (%d, %d)' % (out.shape.as_list()[1], output_shape))

            w = weight_variable(
                [out.shape.as_list()[1], output_shape], seed=self.seed)
            b = bias_variable([output_shape])
            h = tf.nn.relu(tf.add(tf.matmul(out, w), b))
            out = h
            # out = tf.nn.dropout(out, self.keep_prob)
            out = tf.nn.dropout(out, self.dropout_rate)

            hidden_layer_list.append(out)
            weight_list.append(w)

        with tf.name_scope('hidden_layer6'):
            output_shape = int(out.shape.as_list()[1]*0.5)
            print('shape: (%d, %d)' % (out.shape.as_list()[1], output_shape))

            w = weight_variable(
                [out.shape.as_list()[1], output_shape], seed=self.seed)
            b = bias_variable([output_shape])
            h = tf.nn.relu(tf.add(tf.matmul(out, w), b))
            out = h
            # out = tf.nn.dropout(out, self.keep_prob)
            out = tf.nn.dropout(out, self.dropout_rate)

            hidden_layer_list.append(out)
            weight_list.append(w)

#         with tf.name_scope('hidden_layer7'):
#             output_shape = int(out.shape.as_list()[1]*0.5)
#             print('shape: (%d, %d)' % (out.shape.as_list()[1],output_shape))

#             w = weight_variable([out.shape.as_list()[1], output_shape], seed=self.seed)
#             b = bias_variable([output_shape])
#             h = tf.nn.relu(tf.add(tf.matmul(out, w), b))
#             out = h
#             out = tf.nn.dropout(out, self.keep_prob)

#             hidden_layer_list.append(out)
#             weight_list.append(w)

#         with tf.name_scope('hidden_layer8'):
#             output_shape = int(out.shape.as_list()[1]*0.5)
#             print('shape: (%d, %d)' % (out.shape.as_list()[1],output_shape))

#             w = weight_variable([out.shape.as_list()[1], output_shape], seed=self.seed)
#             b = bias_variable([output_shape])
#             h = tf.nn.relu(tf.add(tf.matmul(out, w), b))
#             out = h
#             out = tf.nn.dropout(out, self.keep_prob)

#             hidden_layer_list.append(out)
#             weight_list.append(w)

#         with tf.name_scope('hidden_layer9'):
#             output_shape = int(out.shape.as_list()[1]*0.5)
#             print('shape: (%d, %d)' % (out.shape.as_list()[1],output_shape))

#             w = weight_variable([out.shape.as_list()[1], output_shape], seed=self.seed)
#             b = bias_variable([output_shape])
#             h = tf.nn.relu(tf.add(tf.matmul(out, w), b))
#             out = h
#             out = tf.nn.dropout(out, self.keep_prob)

#             hidden_layer_list.append(out)
#             weight_list.append(w)

        with tf.name_scope('output_layer'):
            output_shape = self.y_out.shape.as_list()[1]
            print('shape: (%d, %d)' % (out.shape.as_list()[1], output_shape))

            w = weight_variable(
                [out.shape.as_list()[1], output_shape], seed=self.seed)
            b = bias_variable([output_shape])
#             h = tf.nn.relu(tf.add(tf.matmul(out, w), b))
            h = tf.add(tf.matmul(out, w), b)
            out = h
            # out = tf.nn.dropout(out, self.keep_prob)

            hidden_layer_list.append(out)
            weight_list.append(w)

        return hidden_layer_list, weight_list

    def loss_op(self):
        with tf.name_scope('cost'):
            self.output_logits = self.h[-1]
            # self.output_logits_split_list = tf.split(self.output_logits, [2,1], axis=1)
            # self.y_out_split_list = tf.split(self.y_out, [2,1], axis=1)
            # print(self.y_out[:2].shape)
            loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.output_logits, labels=self.y_out)
            # loss = tf.losses.sigmoid_cross_entropy(logits=self.output_logits, multi_class_labels=self.y_out)
            # loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_logits_split_list[0], labels=self.y_out_split_list[0]) + \
            #         tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_logits_split_list[1], labels=self.y_out_split_list[1])

            if self.class_weights is not None:
                class_weights = tf.constant(self.class_weights)
            else:
                class_weights = tf.constant([1.0])

            weights = tf.reduce_sum(class_weights * self.y_out, axis=1)
            # self.print_op = tf.print("weights:", weights)
            weighted_losses = loss * weights
            self.cost = tf.reduce_mean(weighted_losses)

            # self.cost = tf.reduce_mean(loss)

            # # if you want to use tf.nn.sigmoid_cross_entropy_with_logits,
            # # remember it'll return loss in class-wise,
            # # so you should take mean of classes first, then take mean of batch
            # loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.output_logits, labels=self.y_out)
            # self.cost = tf.reduce_mean(tf.reduce_mean(loss, axis=1))

    def print_cost(self, feed_dict):
        feed_dict2 = {}
        feed_dict2[self.x_in] = feed_dict['x']
        feed_dict2[self.y_out] = feed_dict['y']
        feed_dict2[self.keep_prob] = feed_dict['k']

        result = self.sess.run([self.cost, self.output_label_index,
                                self.output_multi_label_class], feed_dict=feed_dict2)
        # print(result.shape)

        print("\nprint result...")
        for value in result:
            print("shape: ", value.shape)
            print("value: ", value)

    def predict_op(self):
        with tf.name_scope('output_proba'):
            #self.output_proba = tf.concat([tf.nn.softmax(self.output_logits_split_list[0]), tf.nn.sigmoid(self.output_logits_split_list[1])], axis=1)
            self.output_proba = tf.nn.softmax(self.output_logits)
#             self.output_proba = tf.nn.sigmoid(self.output_logits)

        with tf.name_scope('output_label_index'):
            self.output_label_index = tf.argmax(self.output_proba, axis=1)

        with tf.name_scope('output_multi_label_class'):
            # suppose r is 3-classes multi-label probability
            r = self.output_proba
            # print(r.shape)

            # set 3 different threshold for each class [0.5, 0.6, 0.3]
            if self.class_threshold is not None:
                cond = tf.greater(r, tf.constant(self.class_threshold))
                # print(cond.shape)
            else:
                cond = tf.greater(r, tf.constant([0.5]))

            # return 0 for condition is true (greater then threshold) or 1 for false (less equal then threshold)
            result = tf.where(cond, tf.fill(tf.shape(r), 1),
                              tf.fill(tf.shape(r), 0))
            # print(result.shape)

            self.output_multi_label_class = result

    def metrics_op(self):
        with tf.name_scope('auc'):
            self.auc = tf.metrics.auc(
                labels=self.y_out, predictions=self.output_proba)

        with tf.name_scope('accuracy'):
            correct_prediction = tf.equal(
                tf.argmax(self.output_proba, axis=1), tf.argmax(self.y_out, axis=1))
            self.compute_acc = tf.reduce_mean(
                tf.cast(correct_prediction, dtype=tf.float32))

    def train_op(self):
        with tf.name_scope('train_step'):
            # self.train_step = tf.train.AdamOptimizer(0.001).minimize(self.cost)
            # self.train_step = tfa.optimizers.RectifiedAdam(lr=1e-3).minimize(self.cost)
#             from keras_radam.training import RAdamOptimizer
#             from lib.lookahead import LookaheadOptimizer

            opt = tf.train.AdamOptimizer(0.001)
#             opt = RAdamOptimizer(learning_rate=1e-2)
#             opt = LookaheadOptimizer(opt)
            # self.train_step = RAdamOptimizer(learning_rate=1e-3).minimize(self.cost)
            self.train_step = opt.minimize(self.cost)
