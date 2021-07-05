from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import numpy as np
import tensorflow.compat.v1 as tf1
import sklearn.metrics as skm
import matplotlib.pyplot as plt

from ProgressBar import *
from global_param import *
import model_util as model_util
import math
import resnet
import data_loader as data_loader
import model_loader as model_loader


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
  """
  This function prints and plots the confusion matrix.
  Normalization can be applied by setting `normalize=True`.
  """
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print("Normalized confusion matrix")
  else:
    print('Confusion matrix, without normalization')

  print(cm)

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  plt.xticks(np.arange(len(classes)), classes)
  plt.yticks(np.arange(len(classes)), classes)

  fmt = '.2f' if normalize else 'd'
  thresh = cm.max() / 2.
  for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
      plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")

  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

data_fpr = []
data_tpr = []
def plot_roc_curve(label, score):
  fpr, tpr, thresholds = skm.roc_curve(label, score)
  data_fpr.append(fpr)
  data_tpr.append(tpr)
  plt.plot(fpr, tpr, marker='.')
  plt.show()


def offline_metrics(logits, labels):
  '''

  :param logits:
  :param labels:
  :return: return multiple metrics of binary classification results.
  '''
  def sigmoid(x):
    return 1 / (1 + np.exp(-x))
  score = sigmoid(logits)
  pred = (score > 0.5).astype('int32')
  cm = skm.confusion_matrix(labels, pred)
  wrong_idx = np.squeeze(np.argwhere(pred != labels))
  tn, fp, fn, tp = cm.flatten()
  sensitivity = tp / (tp + fn)
  specifity = tn / (tn + fp)
  precision = tp / (tp + fp)
  recall = sensitivity
  f1_score = 2 * (precision * recall) / (precision + recall)
  plot_confusion_matrix(cm, ['neg', 'pos'])
  plot_roc_curve(labels, score)
  auc = skm.roc_auc_score(labels, score)

  # find optimum threshold
  _, _, thresholds = skm.roc_curve(labels, score)
  max_f1_score = -1
  optimum_threshold = 0
  for th in thresholds:
    _pred = (score > th).astype('int32')
    _cm = skm.confusion_matrix(labels, _pred)
    _tn, _fp, _fn, _tp = _cm.flatten()
    if (_tp + _fp) == 0:
      continue
    _precision = (_tp / (_tp + _fp))
    _recall = _tp / (_tp + _fn)
    _f1_score = 2 * (_precision * _recall) / (_precision + _recall)
    if _f1_score > max_f1_score:
      max_f1_score = _f1_score
      optimum_threshold = th
  print("optimum_threshold:{}, max_f1_score:{}".format(optimum_threshold, max_f1_score))

  return sensitivity, specifity, max_f1_score, wrong_idx, auc


heatmap_count = 0
def heatmap(inputs, ground_truth, logits, ff_map, grads):
  '''

  :param inputs:
  :param ground_truth:
  :param logits:
  :param ff_map:
  :param grads:
  :return: generate and output the heatmap of resulting images
  '''
  global heatmap_count
  weights = np.mean(grads, axis=(1, 2))
  cam = np.einsum('bijk,bk->bij', ff_map, weights)
  cam = np.maximum(cam, 0)
  cam = cam / np.max(cam, axis=(1, 2), keepdims=True)

  import cv2
  import matplotlib.pyplot as plt

  colored_maps = []
  # generate heatmap
  for i in range(cam.shape[0]):
    map = cam[i, :, :]
    map = cv2.resize(map, dsize=(224, 224), interpolation=cv2.INTER_CUBIC)
    map = np.clip(map, 0., 1.)
    fig = plt.figure(figsize=(FLAGS.image_size / 100, FLAGS.image_size / 100))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.imshow(map, cmap='jet')
    fig.canvas.draw()
    colored_cam = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    colored_cam = colored_cam.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    colored_maps.append(colored_cam)

  # overlap heatmaps over input images
  for i in range(inputs.shape[0]):
    # only for predicted positive samples
    if logits[i] < 0:
      continue
    merged = (inputs[i] * 255. * 0.6 + colored_maps[i] * 0.4).astype('uint8')

    fig = plt.figure(figsize=(FLAGS.image_size / 72, FLAGS.image_size / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    heatmap_count += 1
    ax.imshow(merged)
    plt.savefig(FLAGS.model_dir + r'/heatmap/heatmap_%s_%s.jpg' % (heatmap_count, ground_truth[i][0]))
    ax.imshow((inputs[i] * 255).astype('uint8'))
    plt.savefig(FLAGS.model_dir + r'/heatmap/input_%s_%s.jpg' % (heatmap_count, ground_truth[i][0]))

    plt.close()


def perform_evaluation(num_train_examples, eval_steps, model, num_classes,
                       checkpoint_path):
  """Perform evaluation.

    Args:
      num_train_examples: num of training examples
      eval_steps: Number of steps for evaluation.
      model: Instance of transfer_learning.models.Model.
      num_classes: Number of classes to build model for.
      checkpoint_path: Path of checkpoint to be evaluated.

    Returns:
      result: A Dict of metrics and their values.
    """
  result = {}
  graph_eval = tf1.Graph()
  logits = np.empty((0, num_classes))
  labels = np.empty((0, num_classes))
  imgs = np.empty(shape=(0, 224, 224, 3))
  with graph_eval.as_default():
    eval_dataset_gen_fn = data_loader.build_input_fn(False)
    eval_images, eval_labels, eval_iterator = eval_dataset_gen_fn({'batch_size': FLAGS.eval_batch_size})
    model_fn = model_loader.build_model_fn(model, num_classes, num_train_examples)
    eval_loss, metric_fn, metrics = model_fn(
      eval_images, eval_labels, 'eval', params={'batch_size': FLAGS.train_batch_size}
    )
    global_step = tf1.train.get_or_create_global_step()
    metric_op = metric_fn(**metrics)
    eval_loss_mean, eval_loss_mean_update = tf1.metrics.mean(eval_loss)
    init_metric = tf1.local_variables_initializer()
    saver = tf1.train.Saver(max_to_keep=FLAGS.keep_checkpoint_max)

    ff_map = tf1.get_default_graph().get_tensor_by_name('base_model/final_feature_map:0')  # (b,7,7,2048)
    grad = tf1.gradients(metrics['logits_sup'], ff_map)[0]  # (b,7,7,2048)
    norm_grad = tf1.div(grad, tf1.sqrt(tf1.reduce_mean(tf1.square(grad))) + tf1.constant(1e-8))  # (b,7,7,2048)

    with tf1.Session(graph=graph_eval) as sess:
      sess.run(tf1.global_variables_initializer())
      saver.restore(sess, checkpoint_path)
      sess.run(init_metric)
      sess.run(eval_iterator.initializer)
      for _ in range(eval_steps):
        try:
          results = sess.run([metrics['logits_sup'],
                              metrics['labels_sup'],
                              eval_images,
                              ff_map,
                              norm_grad,
                              eval_loss_mean_update] + [op[1] for op in metric_op.values()])
          logits = np.concatenate([logits, results[0]])
          labels = np.concatenate([labels, results[1]])
          imgs = np.concatenate([imgs, results[2]], axis=0)
          # output heatmap
          if FLAGS.output_heatmap:
            heatmap(results[2], results[1], results[0], results[3], results[4])
        except tf1.errors.OutOfRangeError:
          break
      logits = np.squeeze(logits)
      labels = np.squeeze(labels)
      # get metrics result
      result['global_step'] = sess.run(global_step)
      for key, metrics_and_update_op in metric_op.items():
        result[key] = sess.run(metrics_and_update_op[0])
      if num_classes == 1:
        sensitivity, specifity, f1_score, wrong_idx, auc = offline_metrics(logits, labels)
        print('wrong labels:')
        print(labels[wrong_idx])
        print('wrong logits:')
        print(logits[wrong_idx])
        result['sensitivity'] = sensitivity
        result['specifity'] = specifity
        result['max_f1_score'] = f1_score
        result['auc'] = auc
  # Record results as JSON.
  result_json_path = os.path.join(FLAGS.model_dir, 'result.json')
  with tf1.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  result_json_path = os.path.join(
    FLAGS.model_dir, 'result_%d.json' % result['global_step'])
  with tf1.io.gfile.GFile(result_json_path, 'w') as f:
    json.dump({k: float(v) for k, v in result.items()}, f)
  flag_json_path = os.path.join(FLAGS.model_dir, 'flags.json')
  with tf1.io.gfile.GFile(flag_json_path, 'w') as f:
    json.dump(FLAGS.flag_values_dict(), f)

  return result


def train():
  print(FLAGS.flag_values_dict())

  # Enable training summary.
  if FLAGS.train_summary_steps > 0:
    tf1.config.set_soft_device_placement(True)

  num_classes = FLAGS.num_classes
  train_steps = model_util.get_train_steps(FLAGS.num_train_examples)
  eval_steps = int(math.ceil(FLAGS.num_eval_examples / FLAGS.eval_batch_size))
  epoch_steps = int(round(FLAGS.num_train_examples / FLAGS.train_batch_size))

  resnet.BATCH_NORM_DECAY = FLAGS.batch_norm_decay
  model = resnet.resnet_v1(
    resnet_depth=FLAGS.resnet_depth,
    width_multiplier=FLAGS.width_multiplier,
    cifar_stem=FLAGS.image_size <= 32)

  if FLAGS.mode == 'eval':
    for ckpt in tf1.train.checkpoints_iterator(FLAGS.model_dir, min_interval_secs=15):
      result = perform_evaluation(FLAGS.num_train_examples, eval_steps, model, num_classes, ckpt)
      print(result)
  else:
    graph = tf1.Graph()
    with graph.as_default():
      with tf1.variable_scope(name_or_scope='', reuse=tf1.AUTO_REUSE):
        train_dataset_gen_fn = data_loader.build_input_fn(True)
        train_images, train_labels, train_iterator = train_dataset_gen_fn({'batch_size': FLAGS.train_batch_size})
        model_fn = model_loader.build_model_fn(model, num_classes, FLAGS.num_train_examples)
        train_op, loss, variables_to_train, learning_rate, logits_sup = model_fn(
          train_images, train_labels, 'train', params={'batch_size': FLAGS.train_batch_size}
        )

        if FLAGS.mode == 'train_then_eval':
          eval_dataset_gen_fn = data_loader.build_input_fn(False)
          eval_images, eval_labels, eval_iterator = eval_dataset_gen_fn({'batch_size': FLAGS.eval_batch_size})
          eval_loss, metric_fn, metrics = model_fn(
            eval_images, eval_labels, 'eval', params={'batch_size': FLAGS.train_batch_size}
          )
          metric_op = metric_fn(**metrics)

        extra_update_ops = tf1.get_collection(tf1.GraphKeys.UPDATE_OPS)  # update batchnorm status

        if FLAGS.checkpoint:
          tf1.train.init_from_checkpoint(
            FLAGS.checkpoint,
            {v.op.name: v.op.name
             for v in tf1.global_variables(FLAGS.variable_schema)})
          if FLAGS.zero_init_logits_layer:
            # Init op that initializes output layer parameters to zeros.
            output_layer_parameters = [
              var for var in tf1.trainable_variables() if var.name.startswith('head_supervised')]
            tf1.logging.info('Initializing output layer parameters %s to zero',
                            [x.op.name for x in output_layer_parameters])
            zero_init_op = tf1.group([
              tf1.assign(x, tf1.zeros_like(x))
              for x in output_layer_parameters])
          else:
            zero_init_op = None

        train_summaries = []
        train_summaries.append(tf1.summary.scalar('train_loss', loss))
        train_summaries.append(tf1.summary.scalar('learning_rate', learning_rate))
        train_top_1_accuracy, train_top_1_accuracy_update = tf1.metrics.accuracy(
          tf1.argmax(train_labels['labels'], 1), tf1.argmax(logits_sup, axis=1)
        ) if num_classes > 1 else tf1.metrics.accuracy(
          train_labels['labels'], tf1.where(tf1.less(tf1.nn.sigmoid(logits_sup), 0.5), tf1.zeros_like(logits_sup),
                                           tf1.ones_like(logits_sup)))
        train_summaries.append(tf1.summary.scalar('train_top_1_accuracy', train_top_1_accuracy))

        eval_summaries = []
        if FLAGS.mode == 'train_then_eval':
          eval_loss_mean, eval_loss_mean_update = tf1.metrics.mean(eval_loss)
          eval_summaries.append(tf1.summary.scalar('eval_loss', eval_loss_mean))
          for metric_name, metric_and_update_op in metric_op.items():
            eval_summaries.append(tf1.summary.scalar(metric_name, metric_and_update_op[0]))

        init_metric = tf1.local_variables_initializer()

        train_merge_summary = tf1.summary.merge(train_summaries)
        if FLAGS.mode == 'train_then_eval':
          eval_merge_summary = tf1.summary.merge(eval_summaries)

        saver = tf1.train.Saver(max_to_keep=FLAGS.keep_checkpoint_max)

      with tf1.Session(graph=graph) as sess:
        sess.run(tf1.global_variables_initializer())
        if zero_init_op:
          sess.run(zero_init_op)
        sess.run(init_metric)
        global_step = tf1.train.get_or_create_global_step()
        writer = tf1.summary.FileWriter(FLAGS.model_dir, sess.graph)
        bar = ProgressBar(sess.run(global_step))
        while sess.run(global_step) < train_steps:
          for _ in range(epoch_steps):
            try:
              sess.run([train_op, extra_update_ops, train_top_1_accuracy_update])
              bar.step_forward()
            except tf1.errors.OutOfRangeError:
              break
          if (sess.run(global_step) // epoch_steps) % FLAGS.checkpoint_epochs == 0:
            saver.save(sess, FLAGS.model_dir + '/model', global_step=global_step)
          # write summary
          summary = sess.run(train_merge_summary)
          writer.add_summary(summary=summary, global_step=sess.run(global_step))

          sess.run(init_metric)
          if FLAGS.mode == 'train_then_eval':
            sess.run(eval_iterator.initializer)
            for _ in range(eval_steps):
              try:
                sess.run([eval_loss_mean_update] + [op[1] for op in metric_op.values()])
              except tf1.errors.OutOfRangeError:
                break
            # write summary
            summary = sess.run(eval_merge_summary)
            writer.add_summary(summary=summary, global_step=sess.run(global_step))

        saver.save(sess, FLAGS.model_dir + '/model', global_step=global_step)
        writer.close()

  return 0


def main(argv):
  train()


if __name__ == '__main__':
  argv = ['run.py',
          '--learning_rate=5e-4',
          '--warmup_epochs=20',
          '--weight_decay=1e-4',
          '--train_batch_size=32',
          '--train_epochs=400',
          '--eval_batch_size=32',
          '--train_summary_steps=0',
          '--mode=eval',
          '--train_mode=finetune',
          '--checkpoint=./tmp/pretrained',
          '--variable_schema=(?!global_step|(?:.*/|^)Momentum|(?:.*/|^)Adam|beta1|beta2|head_supervised)',
          '--zero_init_logits_layer=False',
          '--fine_tune_after_block=4',
          '--model_dir=./tmp/tuberculosis',
          '--data_dir=../../datasets/medical_images/tuberculosis',
          '--temperature=0.1',
          '--ft_proj_selector=0',
          '--width_multiplier=1',
          '--resnet_depth=50',
          '--sk_ratio=0.0625',
          '--image_size=224',
          '--color_jitter_strength=0.5',
          '--use_blur=True',
          '--crop_area_lower_bound=0.65',
          '--use_learning_rate_schedule=True',
          '--finetune_projection_head=False',
          '--checkpoint_epochs=10',
          '--keep_checkpoint_max=2',
          '--dropout_rate=0.0',
          '--num_classes=1',
          '--num_train_examples=446',
          '--num_remain_examples=0',
          '--num_eval_examples=160',
          '--crop_strategy=centerized']
  app.run(main, argv=argv)