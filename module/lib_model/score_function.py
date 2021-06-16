#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /media/allenyl/DATA/Projects/server_setup/repo-ai/NLP-AI-in-Law/samples/score_function.py
# Project: /media/allenyl/DATA/Projects/server_setup/repo-ai/NLP-AI-in-Law/samples
# Created Date: Tuesday, December 18th 2018, 11:50:52 am
# Author: Allenyl(allen7575@gmail.com>)
# -----
# Last Modified: Thursday, July 2nd 2020, 1:29:07 pm
# Modified By: Allenyl(allen7575@gmail.com)
# -----
# Copyright 2018 - 2018 Allenyl Copyright, Allenyl Company
# -----
# license:
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
# ------------------------------------
# HISTORY:
# Date      	By	Comments
# ----------	---	---------------------------------------------------------
###
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import copy

def print_score(model, x, y_true, case_id_list=None, verbose=False, show_graph=True, only_use_proba=True, mis_classify_list=None, total_list=None, threshold_list=None):

    # if not binary, use one-hot encode
    if type(y_true[0]) != np.ndarray:
        n_classes = len(np.unique(y_true))
        y_true = np.eye(len(y_true),n_classes)[y_true]

    n_classes = y_true.shape[1]

    if only_use_proba:
        if threshold_list is not None:
            y_pred_proba = model.predict_proba(x, threshold_list=threshold_list)
        else:
            y_pred_proba = model.predict_proba(x)
            # print(y_pred_proba)

        label_index = np.argmax(y_pred_proba, axis=1)

        label = np.zeros(shape=(len(label_index), n_classes))
        label[np.arange(len(label_index)),label_index]=1

        y_pred = label
        # print(y_pred)

    else:
        if threshold_list is not None:
            y_pred = model.predict(x, threshold_list=threshold_list)
            y_pred_proba = model.predict_proba(x, threshold_list=threshold_list)
        else:
            y_pred = model.predict(x)
            y_pred_proba = model.predict_proba(x)

        # if not binary, use one-hot encode
        if type(y_pred[0]) != np.ndarray:
            y_pred = np.eye(len(y_pred),n_classes)[y_pred]

    if np.any(np.isnan(y_pred_proba)):
        print("diverged!")
        return np.nan, np.nan, np.nan, np.nan

    if total_list is not None:
        for i in range(len(y_pred)):
            if case_id_list is None:
                case_id = i
            else:
                case_id = case_id_list[i]

            log_dict = {
                'ID': case_id,
                'predict': y_pred[i].astype(int),
                'truth': y_true[i],
                'proba': y_pred_proba[i]
            }

            total_list.append(log_dict)

            if mis_classify_list is not None:
                if not np.array_equal(y_pred[i], y_true[i]):
                    # print(np.argmax(y_pred[i]))
                    delta = y_pred_proba[i][np.argmax(y_pred[i])] - y_pred_proba[i][np.argmax(y_true[i])]

                    mis_classify_list.append(log_dict)
                    if verbose:
                        print(
                            "mis classified:", case_id, "pred:", y_pred[i].astype(int),
                            "truth:", y_true[i], "proba:", y_pred_proba[i], "\tdelta:",
                            delta
                        )

    # number of classes
    # try:
    #     n_classes = y_pred.shape[1]
    # except IndexError:
    #     n_classes = 0
    #     y_pred_proba = y_pred_proba.T[1]

    # try:
    #     y_pred.shape[1]
    # except IndexError:
    #     y_test_one_hot = np.zeros((len(y_pred),2))
    #     y_test_one_hot[np.arange(len(y_pred)),y_pred]=1
    #     y_pred = y_test_one_hot

    unpredicted_label = set(np.argmax(y_true, axis=1)) - set(np.argmax(y_pred, axis=1))

    if show_graph:
        # print('\n\n')
        print('n_classes: ', n_classes)

        if threshold_list is not None:
            print('threshold:', np.array(threshold_list).round(3))

        ###### accuracy ######
        from sklearn.metrics import accuracy_score

        # we use model.predict to get the label
        # print(y_true)
        # print(y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        ###### precision ######
        from sklearn.metrics import precision_score

        # get precision score
        score = precision_score(y_true, y_pred, average=None)
        for i in range(len(score)):
            print("class %d Precision: %.2f%%" % (i, score[i] * 100.0))

        ###### recall ######
        from sklearn.metrics import recall_score
        # get recall score
        score = recall_score(y_true, y_pred, average=None)
        for i in range(len(score)):
            print("class %d Recall: %.2f%%" % (i, score[i] * 100.0))

        ###### F1 ######
        from sklearn.metrics import f1_score
        # get F1 score
        if len(unpredicted_label):
            print("unpredicted label(s): {}".format(unpredicted_label))

        f1_value = f1_score(y_true, y_pred, average=None)
        for i in range(len(f1_value)):
            print("class %d F1: %.2f%%" % (i, f1_value[i] * 100.0))

        f1_value = np.mean(f1_value)

        ###### confusion matrix ######
        from sklearn.metrics import confusion_matrix
        # get confusion matrix

        if n_classes == 0:
            confu_matrix = confusion_matrix(y_true, y_pred)
        else:
            # print(y_true.shape)
            # print(y_pred.shape)
            # [python - Multilabel-indicator is not supported for confusion matrix - Stack Overflow](https://stackoverflow.com/questions/46953967/multilabel-indicator-is-not-supported-for-confusion-matrix)
            confu_matrix = confusion_matrix(y_true.argmax(axis=1), y_pred.argmax(axis=1))
        print('confusion_matrix: \n' , confu_matrix)

        ##########################################################
        ###### ROC(Receiver Operating Characteristic) curve ######
        ##########################################################
        from sklearn import metrics

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        thresholds = dict()
        roc_auc = dict()

        if n_classes == 0:
            i = 0
            fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y_true[:], y_pred_proba[:], pos_label=1)
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        else:
            for i in range(n_classes):
                fpr[i], tpr[i], thresholds[i] = metrics.roc_curve(y_true[:, i], y_pred_proba[:, i], pos_label=1)
                roc_auc[i] = metrics.auc(fpr[i], tpr[i])


        # Plot of a ROC curve for a specific class
        plt.figure()

        for i in range(len(roc_auc)):
            line, = plt.plot(fpr[i], tpr[i], label='class %d (area = %0.2f)' % (i, roc_auc[i]))
            # plot threshold
            plt.plot(fpr[i][1:], thresholds[i][1:], ':', color=line.get_color())
            # print(fpr[i][1:], thresholds[i][1:])
        # legend for threshold
        plt.plot([0], [0], 'k:', label='threshold')

        # plot 0.5 roc-auc line
        plt.plot([0, 1], [0, 1], 'k--')
        # 0.5 TPR line
        plt.plot([0, 1], [0.5, 0.5], 'k--')

        # plot axis
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")

        # plot y-axis label and tick on right
        ax = plt.gca().twinx()
        plt.ylabel('threshold')
        plt.ylim([-0.05, 1.05])
        plt.show()


        ###### get roc-auc value without calculate roc ######
        from sklearn.metrics import roc_auc_score

        # predict() returns only one class or the other.
        # Then you compute a ROC with the results of predict on a classifier,
        # there are only three thresholds (trial all one class, trivial all the other class, and in between).
        print('label roc-auc: ', roc_auc_score(y_true, y_pred, average='macro'))

        # predict_proba() returns an entire range of probabilities,
        # so now you can put more than three thresholds on your data.
        roc_auc = roc_auc_score(y_true, y_pred_proba, average='macro')
        print('probability roc-auc: ', roc_auc)


        ########################################
        ###### PR(Precision Recall) curve ######
        ########################################
        from sklearn import metrics

        # Compute PR curve and PR area for each class
        precision = dict()
        recall = dict()
        thresholds = dict()
        pr_auc = dict()

        if n_classes == 0:
            i = 0
            precision[i], recall[i], thresholds[i] = metrics.precision_recall_curve(y_true[:], y_pred_proba[:], pos_label=1)
            pr_auc[i] = metrics.auc(recall[i], precision[i])
        else:
            for i in range(n_classes):
                precision[i], recall[i], thresholds[i] = metrics.precision_recall_curve(y_true[:, i], y_pred_proba[:, i], pos_label=1)
                pr_auc[i] = metrics.auc(recall[i], precision[i])


        # Plot of a ROC curve for a specific class
        plt.figure()

        for i in range(len(pr_auc)):
            # plot pr curve
            line, = plt.plot(recall[i], precision[i], label='class %d (area = %0.2f)' % (i, pr_auc[i]))
            # plot threshold
            plt.plot(recall[i][1:], thresholds[i][:], ':', color=line.get_color())

        # legend for threshold
        plt.plot([0], [0], 'k:', label='threshold')

        # plot iso-f1 cruve
        for f1_value_tmp in np.arange(0.1, 1, 0.2):
            recall_list = np.arange(f1_value_tmp/(2-f1_value_tmp), 1, 0.001)
            precision_list = []
            for i in range(len(recall_list)):
                precision_list.append(1/(2/f1_value_tmp - 1/recall_list[i]))
            plt.plot(recall_list, precision_list, 'r--')
            position_index = round(len(recall_list)/5)
            position = (recall_list[position_index], precision_list[position_index])
            plt.annotate('F1={:.1f}'.format(f1_value_tmp), position)

        # legend for iso-F1 curve
        plt.plot([0], [0], 'r--', label='iso-F1')

        # 0.5 precision line
        plt.plot([0, 1], [0.5, 0.5], 'k--')

        # plot axis
        plt.xlim([-0.05, 1.0])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")

        # plot y-axis label and tick on right
        ax = plt.gca().twinx()
        plt.ylabel('threshold')
        plt.ylim([-0.05, 1.05])
        plt.show()


        ###### get pr-auc value without calculate pr curve ######
        from sklearn.metrics import average_precision_score

        # predict() returns only one class or the other.
        # Then you compute a ROC with the results of predict on a classifier,
        # there are only three thresholds (trial all one class, trivial all the other class, and in between).
        print('label pr-auc: ', average_precision_score(y_true, y_pred, average='macro'))

        # predict_proba() returns an entire range of probabilities,
        # so now you can put more than three thresholds on your data.
        pr_auc = average_precision_score(y_true, y_pred_proba, average='macro')
        print('probability pr-auc: ', pr_auc)

    else:
        ###### metrics ######
        from sklearn.metrics import accuracy_score
        from sklearn.metrics import roc_auc_score
        from sklearn.metrics import average_precision_score
        from sklearn.metrics import f1_score

        accuracy = accuracy_score(y_true, y_pred)
        # print(y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba, average='macro')
        pr_auc = average_precision_score(y_true, y_pred_proba, average='macro')

        if len(unpredicted_label):
            print("unpredicted label(s): {}".format(unpredicted_label))

        f1_value = f1_score(y_true, y_pred, average=None)
        f1_value = np.mean(f1_value)

        # if len(set(np.argmax(y_true, axis=1)) - set(np.argmax(y_pred, axis=1))) != 0:
        #     # prevent warning: "F-score is ill-defined and being set to 0.0 in labels with no predicted samples."
        #     # https://stackoverflow.com/questions/43162506/undefinedmetricwarning-f-score-is-ill-defined-and-being-set-to-0-0-in-labels-wi
        #     f1_value = 0
        # else:
        #     f1_value = f1_score(y_true, y_pred, average='macro')

    return accuracy, roc_auc, pr_auc, f1_value



class proxyModel():
    def __init__(self, origin_model):
        self.origin_model = origin_model

    def predict_proba(self, x, threshold_list=None):
        # get origin probability
        ori_proba = self.origin_model.predict_proba(x)

        # set default threshold
        if threshold_list is None:
            threshold_list = np.full(ori_proba[0].shape, 1)

        # get the output shape of threshold_list
        output_shape = np.array(threshold_list).shape

        # element-wise divide by the threshold of each classes
        new_proba = np.divide(ori_proba, threshold_list)

        # calculate the norm (sum of new probability of each classes)
        norm = np.linalg.norm(new_proba, ord=1, axis=1)

        # reshape the norm
        norm = np.broadcast_to(np.array([norm]).T, (norm.shape[0],output_shape[0]))

        # renormalize the new probability
        new_proba = np.divide(new_proba, norm)

        return new_proba

    def predict(self, x, threshold_list=None):
        return np.argmax(self.predict_proba(x, threshold_list), axis=1)



def weighted_score_with_threshold(threshold, model, X_test, Y_test, metrics='accuracy', delta=5e-5):
    # if the sum of thresholds were not between 1+delta and 1-delta,
    # return infinity (just for reduce the search space of the minimizaiton algorithm,
    # because the sum of thresholds should be as close to 1 as possible).
    threshold_sum = np.sum(threshold)

    if threshold_sum > 1+delta:
        return np.inf

    if threshold_sum < 1-delta:
        return np.inf

    # to avoid objective function jump into nan solution
    if np.isnan(threshold_sum):
        # print("threshold_sum is nan")
        return np.inf

    # renormalize: the sum of threshold should be 1
    normalized_threshold = threshold/threshold_sum

    # calculate score based on thresholds
    # accuracy, roc_auc, pr_auc, f1 = print_score(model, X_test, Y_test, show_graph=False, only_use_proba=False, threshold_list=normalized_threshold)
    scores = print_score(model, X_test, Y_test, show_graph=False, threshold_list=normalized_threshold)

    scores = np.array(scores)
    weight = np.array([1,1,1,1])

    # Give the metric you want to maximize a bigger weight:
    if metrics == 'accuracy':
        weight = np.array([10,1,1,1])
    elif metrics == 'roc_auc':
        weight = np.array([1,10,1,1])
    elif metrics == 'pr_auc':
        weight = np.array([1,1,10,1])
    elif metrics == 'f1':
        weight = np.array([1,1,1,10])
    elif 'all':
        weight = np.array([1,1,1,1])

    # return negatitive weighted sum (because you want to maximize the sum,
    # it's equivalent to minimize the negative sum)
    return -np.dot(weight, scores)



def find_optimal_threshold(proxyModel, X_test, Y_test, objectiveFunc=weighted_score_with_threshold, metrics='accuracy', number_of_tries=20):
    import random
    from scipy import optimize
    from tqdm import tqdm

    output_class_num = Y_test.shape[1]
    bounds = optimize.Bounds([1e-5]*output_class_num,[1]*output_class_num)

    for i in tqdm(range(number_of_tries)):
        seed = random.randrange(2**32 - 1)
        # print(seed)
        # seed = 4089185835
        minimum = optimize.differential_evolution(objectiveFunc, bounds, args=(proxyModel, X_test, Y_test, metrics), seed=seed)

        if i == 0:
            result = minimum
            result_seed = seed
            best_iter = i
            # print("seed:", result_seed)
        elif result.fun > minimum.fun:
            result = minimum
            result_seed = seed
            best_iter = i
            # print("iter:", i, "better_seed:", result_seed)


    # calculate threshold
    threshold = result.x/np.sum(result.x)
    print(f'The best result achieved at {best_iter}th iter, with seed {result_seed}' )

    return threshold




def print_grid_search_map(gridCV_model, parameters_dict, score_string='mean_test_neg_log_loss'):
    # import matplotlib.pyplot as plt
    # %matplotlib inline

    # switch to default_backend to avoid subsequent dynamic plot clear all images
    if matplotlib.get_backend() == 'nbAgg':
        default_backend = 'module://ipykernel.pylab.backend_inline'
        plt.switch_backend(default_backend)
        # print(matplotlib.get_backend())

    params = copy.deepcopy(parameters_dict)
    keys = list(params.keys())
    # print(keys)

    param1 = params.pop(keys[0])
    param2 = params.pop(keys[1])

    # fetch scores, reshape into a grid
    scores = [x for x in gridCV_model.cv_results_[score_string]]
    scores = np.array(scores).reshape(len(param1), len(param2))
    scores = np.transpose(scores)

    # Make heatmap from grid search results
    plt.figure(figsize=(12, 6))
    plt.imshow(scores, interpolation='nearest', origin='higher', cmap='jet_r')
    plt.xticks(np.arange(len(param1)), param1)
    plt.yticks(np.arange(len(param2)), param2)
    plt.xlabel(keys[0])
    plt.ylabel(keys[1])
    plt.colorbar().set_label('neg log loss', rotation=270, labelpad=20)
    plt.show()
