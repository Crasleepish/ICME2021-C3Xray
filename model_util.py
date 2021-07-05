"""Network architectures related functions used in SimCLR."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags

import resnet

import tensorflow.compat.v1 as tf1

FLAGS = flags.FLAGS

LARGE_NUM = 1e9


def add_supervised_loss(labels, logits, weights, num_classes, **kwargs):
  """Compute loss for model and add it to loss collection."""
  if num_classes > 1:
    return tf1.losses.softmax_cross_entropy(labels, logits, weights, **kwargs)
  else:
    return tf1.losses.sigmoid_cross_entropy(labels, logits, tf1.expand_dims(weights, axis=-1), **kwargs)


def add_contrastive_loss(hidden,
                         hidden_norm=True,
                         temperature=1.0,
                         weights=1.0):
  """Compute loss for model.

  Args:
    hidden: hidden vector (`Tensor`) of shape (bsz, dim).
    hidden_norm: whether or not to use normalization on the hidden vector.
    temperature: a `floating` number for temperature scaling.
    weights: a weighting number or vector.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
  """
  # Get (normalized) hidden1 and hidden2.
  if hidden_norm:
    hidden = tf1.math.l2_normalize(hidden, -1)
  hidden1, hidden2 = tf1.split(hidden, 2, 0)
  batch_size = tf1.shape(hidden1)[0]

  # Gather hidden1/hidden2 across replicas and create local labels.
  hidden1_large = hidden1
  hidden2_large = hidden2
  labels = tf1.one_hot(tf1.range(batch_size), batch_size * 2)
  masks = tf1.one_hot(tf1.range(batch_size), batch_size)

  logits_aa = tf1.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
  logits_aa = logits_aa - masks * LARGE_NUM
  logits_bb = tf1.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
  logits_bb = logits_bb - masks * LARGE_NUM
  logits_ab = tf1.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
  logits_ba = tf1.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

  loss_a = tf1.losses.softmax_cross_entropy(
      labels, tf1.concat([logits_ab, logits_aa], 1), weights=weights)
  loss_b = tf1.losses.softmax_cross_entropy(
      labels, tf1.concat([logits_ba, logits_bb], 1), weights=weights)
  loss = loss_a + loss_b

  return loss, logits_ab, labels


def add_weight_decay(adjust_per_optimizer=True):
  """Compute weight decay from flags."""
  l2_losses = [tf1.nn.l2_loss(v) for v in tf1.trainable_variables()
               if 'batch_normalization' not in v.name and 'bias' not in v.name]
  tf1.losses.add_loss(
      FLAGS.weight_decay * tf1.add_n(l2_losses),
      tf1.GraphKeys.REGULARIZATION_LOSSES)


def get_train_steps(num_examples):
  """Determine the number of training steps."""
  return FLAGS.train_steps or (
      num_examples * FLAGS.train_epochs // FLAGS.train_batch_size + 1)


def learning_rate_schedule(base_learning_rate, num_examples):
  """Build learning rate schedule."""
  global_step = tf1.train.get_or_create_global_step()
  warmup_steps = int(round(
      FLAGS.warmup_epochs * num_examples // FLAGS.train_batch_size))
  half_life_steps = int(round(
      FLAGS.half_life * num_examples // FLAGS.train_batch_size))
  learning_rate = (tf1.to_float(global_step) / int(warmup_steps) * base_learning_rate
                   if warmup_steps else base_learning_rate)

  # Exponential decay learning rate schedule
  learning_rate = tf1.where(
      global_step < warmup_steps, learning_rate,
      base_learning_rate * 0.5 ** ((tf1.to_float(global_step) - tf1.to_float(warmup_steps)) / half_life_steps))

  return learning_rate


def get_optimizer(learning_rate):
  """Returns an optimizer."""
  optimizer = tf1.train.MomentumOptimizer(
    learning_rate, FLAGS.momentum, use_nesterov=True)
  return optimizer


def linear_layer(x,
                 is_training,
                 num_classes,
                 use_bias=True,
                 use_bn=False,
                 name='linear_layer'):
  """Linear head for linear evaluation.

  Args:
    x: hidden state tensor of shape (bsz, dim).
    is_training: boolean indicator for training or test.
    num_classes: number of classes.
    use_bias: whether or not to use bias.
    use_bn: whether or not to use BN for output units.
    name: the name for variable scope.

  Returns:
    logits of shape (bsz, num_classes)
  """
  assert x.shape.ndims == 2, x.shape
  with tf1.variable_scope(name, reuse=tf1.AUTO_REUSE):
    x = tf1.layers.dense(
        inputs=x,
        units=num_classes,
        use_bias=use_bias and not use_bn,
        kernel_initializer=tf1.random_normal_initializer(stddev=.01))
    if use_bn:
      x = resnet.batch_norm_relu(x, is_training, relu=False, center=use_bias)
    x = tf1.identity(x, '%s_out' % name)
  return x


def projection_head(hiddens, is_training, name='head_contrastive'):
  """Head for projecting hiddens fo contrastive loss."""
  with tf1.variable_scope(name, reuse=tf1.AUTO_REUSE):
    mid_dim = hiddens.shape[-1]
    out_dim = FLAGS.proj_out_dim
    hiddens_list = [hiddens]
    if FLAGS.proj_head_mode == 'none':
      pass  # directly use the output hiddens as hiddens.
    elif FLAGS.proj_head_mode == 'linear':
      hiddens = linear_layer(
          hiddens, is_training, out_dim,
          use_bias=False, use_bn=True, name='l_0')
      hiddens_list.append(hiddens)
    elif FLAGS.proj_head_mode == 'nonlinear':
      for j in range(FLAGS.num_proj_layers):
        if j != FLAGS.num_proj_layers - 1:
          # for the middle layers, use bias and relu for the output.
          dim, bias_relu = mid_dim, True
        else:
          # for the final layer, neither bias nor relu is used.
          dim, bias_relu = FLAGS.proj_out_dim, False
        hiddens = linear_layer(
            hiddens, is_training, dim,
            use_bias=bias_relu, use_bn=True, name='nl_%d'%j)
        hiddens = tf1.nn.relu(hiddens) if bias_relu else hiddens
        hiddens_list.append(hiddens)
    else:
      raise ValueError('Unknown head projection mode {}'.format(
          FLAGS.proj_head_mode))
    if FLAGS.train_mode == 'pretrain':
      # take the projection head output during pre-training.
      hiddens = hiddens_list[-1]
    else:
      # for checkpoint compatibility, whole projection head is built here.
      # but you can select part of projection head during fine-tuning.
      hiddens = hiddens_list[FLAGS.ft_proj_selector]
      if FLAGS.finetune_projection_head:
        if len(tf1.get_collection('trainable_variables_inblock_5')) == 0:
          for d in range(FLAGS.ft_proj_selector):
            _trainable_variables = tf1.trainable_variables('head_contrastive/nl_%d' % d)
            for var in _trainable_variables:
              tf1.add_to_collection('trainable_variables_inblock_5', var)

  return hiddens


def supervised_head(hiddens, num_classes, is_training, name='head_supervised'):
  """Add supervised head & also add its variables to inblock collection."""
  with tf1.variable_scope(name):
    # hiddens = linear_layer(hiddens, is_training, 32, use_bias=True, use_bn=True, name='sup_l0')
    # hiddens = tf1.nn.relu(hiddens)
    # logits = linear_layer(hiddens, is_training, num_classes, name='sup_l1')
    if is_training:
      hiddens = tf1.nn.dropout(hiddens, rate=FLAGS.dropout_rate)
    logits = linear_layer(hiddens, is_training, num_classes)
  if len(tf1.get_collection('trainable_variables_inblock_6')) == 0:
    for var in tf1.trainable_variables():
      if var.name.startswith(name):
        tf1.add_to_collection('trainable_variables_inblock_6', var)
  return logits
