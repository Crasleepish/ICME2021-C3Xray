"""Model specification for SimCLR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import data_util as data_util
import model_util as model_util

import tensorflow.compat.v1 as tf1
import tensorflow.compat.v2 as tf2

FLAGS = flags.FLAGS

def build_model_fn(model, num_classes, num_train_examples):
  """Build model function."""
  def model_fn(features, labels, mode, params=None):
    """Build model and optimizer."""
    is_training = mode == "train"

    # Check training mode.
    if FLAGS.train_mode == 'pretrain':
      num_transforms = 2
      if FLAGS.fine_tune_after_block > -1:
        raise ValueError('Does not support layer freezing during pretraining,'
                         'should set fine_tune_after_block<=-1 for safety.')
    elif FLAGS.train_mode == 'finetune':
      num_transforms = 1
    else:
      raise ValueError('Unknown train_mode {}'.format(FLAGS.train_mode))

    # Split channels, and optionally apply extra batched augmentation.
    features_list = tf1.split(
      features, num_or_size_splits=num_transforms, axis=-1)
    if FLAGS.use_blur and is_training and FLAGS.train_mode == 'pretrain':
      features_list = data_util.batch_random_blur(
          features_list, FLAGS.image_size, FLAGS.image_size)
    features = tf1.concat(features_list, 0)  # (num_transforms * bsz, h, w, c)

    # Base network forward pass.
    with tf1.variable_scope('base_model'):
      if FLAGS.train_mode == 'finetune' and FLAGS.fine_tune_after_block >= 4:
        # Finetune just supervised (linear) head will not update BN stats.
        model_train_mode = False
      else:
        # Pretrain or finetuen anything else will update BN stats.
        model_train_mode = is_training
      hiddens = model(features, is_training=model_train_mode)

    # Add head and loss.
    if FLAGS.train_mode == 'pretrain':
      hiddens_proj = model_util.projection_head(hiddens, is_training)
      with tf1.name_scope("train_loss" if is_training else "eval_loss"):
        contrast_loss, logits_con, labels_con = model_util.add_contrastive_loss(
          hiddens_proj,
          hidden_norm=FLAGS.hidden_norm,
          temperature=FLAGS.temperature)
      logits_sup = tf1.zeros([params['batch_size'], num_classes])
    else:
      contrast_loss = tf1.zeros([])
      logits_con = tf1.zeros([params['batch_size'], 10])
      labels_con = tf1.zeros([params['batch_size'], 10])
      hiddens = model_util.projection_head(hiddens, is_training)
      logits_sup = model_util.supervised_head(hiddens, num_classes, is_training)
      with tf1.name_scope("train_loss" if is_training else "eval_loss"):
        model_util.add_supervised_loss(
          labels=labels['labels'],
          logits=logits_sup,
          weights=labels['mask'],
          num_classes=num_classes)

    # Add weight decay to loss.
    with tf1.name_scope("train_loss" if is_training else "eval_loss"):
      model_util.add_weight_decay(adjust_per_optimizer=True)

    if is_training:
      loss = tf1.losses.get_total_loss(scope='train_loss')
    else:
      loss = tf1.losses.get_total_loss(scope='eval_loss')

    if FLAGS.train_mode == 'pretrain':
      variables_to_train = tf1.trainable_variables()
    else:
      collection_prefix = 'trainable_variables_inblock_'
      variables_to_train = []
      for j in range(FLAGS.fine_tune_after_block + 1, 7):
        variables_to_train += tf1.get_collection(collection_prefix + str(j))
      assert variables_to_train, 'variables_to_train shouldn\'t be empty!'

    tf1.logging.info('===============Variables to train (begin)===============')
    tf1.logging.info(variables_to_train)
    tf1.logging.info('================Variables to train (end)================')

    learning_rate = model_util.learning_rate_schedule(
      FLAGS.learning_rate, num_train_examples) \
      if FLAGS.use_learning_rate_schedule else FLAGS.learning_rate

    if is_training:
      optimizer = model_util.get_optimizer(learning_rate)
      train_op = optimizer.minimize(
          loss, global_step=tf1.train.get_or_create_global_step(),
          var_list=variables_to_train)
      return train_op, loss, variables_to_train, learning_rate, logits_sup
    else:
      def metric_fn(logits_sup, labels_sup, logits_con, labels_con, mask,
                    **kws):
        """Inner metric function."""
        metrics = {k: tf1.metrics.mean(v)
                   for k, v in kws.items()}
        if FLAGS.train_mode == 'finetune':
          metrics['label_top_1_accuracy'] = tf1.metrics.accuracy(
              tf1.argmax(labels_sup, 1), tf1.argmax(logits_sup, axis=1)
              ) if num_classes > 1 else tf1.metrics.accuracy(
          labels_sup, tf1.where(tf1.less(tf1.nn.sigmoid(logits_sup), 0.5), tf1.zeros_like(logits_sup),
                               tf1.ones_like(logits_sup)))
        # metrics['label_top_5_accuracy'] = tf.metrics.recall_at_k(
        #     tf.argmax(labels_sup, 1), logits_sup, k=5, weights=mask)
        if FLAGS.train_mode == 'pretrain':
          metrics['contrastive_top_1_accuracy'] = tf1.metrics.accuracy(
            tf1.argmax(labels_con, 1), tf1.argmax(logits_con, axis=1)
            )
        # metrics['contrastive_top_5_accuracy'] = tf.metrics.recall_at_k(
        #     tf.argmax(labels_con, 1), logits_con, k=5, weights=mask)
        return metrics

      metrics = {
        'logits_sup': logits_sup,
        'labels_sup': labels['labels'],
        'logits_con': logits_con,
        'labels_con': labels_con,
        'mask': labels['mask'],
        'contrast_loss': tf1.fill((params['batch_size'],), contrast_loss),
        'regularization_loss': tf1.fill((params['batch_size'],),
                                       tf1.losses.get_regularization_loss(
                                         scope=("train_loss" if is_training else "eval_loss"))),
      }
      return loss, metric_fn, metrics

  return model_fn