from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pathlib
import functools
from absl import flags

import data_util as data_util
import tensorflow.compat.v1 as tf1

FLAGS = flags.FLAGS


def get_preprocess_fn(is_training, is_pretrain):
  """Get function that accepts an image and returns a preprocessed image."""
  return functools.partial(
    data_util.preprocess_image,
    height=FLAGS.image_size,
    width=FLAGS.image_size,
    is_training=is_training,
    color_distort=True,
    crop=True,
    flip=True,
    test_crop=True,
    rotate=(FLAGS.use_rotation and is_pretrain)
  )


def build_input_fn(is_training):
  '''
  :param is_training:
  :return:
  '''

  def _input_fn(params):
    """Inner input function."""
    preprocess_fn_pretrain = get_preprocess_fn(True, is_pretrain=True)
    preprocess_fn_finetune = get_preprocess_fn(is_training, is_pretrain=False)
    num_classes = FLAGS.num_classes

    def map_fn(image, label):
      """Produces multiple transformations of the same batch."""
      if FLAGS.crop_strategy == 'center_specific':
        op_center = tf1.clip_by_value(tf1.random.normal([2], [0.5, 0.5], [0.2, 0.2], dtype='float32'), 0.0, 1.0)
      else:
        op_center = tf1.constant([0.5, 0.5], dtype='float32')
      if FLAGS.train_mode == 'pretrain':
        xs = []
        for _ in range(2):  # Two random transformations
          xs.append(preprocess_fn_pretrain(image, op_center))
        image = tf1.concat(xs, -1)
        label = tf1.zeros([num_classes])
      else:
        image = preprocess_fn_finetune(image, tf1.constant([0.5, 0.5], dtype='float32'))
        label = tf1.cast(label, dtype='int32')
        label = tf1.one_hot(label, num_classes) if num_classes > 1 else tf1.expand_dims(label, axis=-1)
      return image, label, 1.0

    dataset = get_local_dataset(is_training)
    if is_training:
      buffer_multiplier = 10
      dataset = dataset.shuffle(params['batch_size'] * buffer_multiplier, reshuffle_each_iteration=True)
      dataset = dataset.repeat(-1)  # The dataset is repeated indefinitely.
    dataset = dataset.map(map_fn, num_parallel_calls=tf1.data.experimental.AUTOTUNE)
    dataset = dataset.batch(params['batch_size'], drop_remainder=is_training)
    if is_training:
      dataset = pad_to_batch(dataset, params['batch_size'])
      it = tf1.data.make_one_shot_iterator(dataset)
      images, labels, mask = it.get_next()
    else:
      it = tf1.data.make_initializable_iterator(dataset)
      images, labels, mask = it.get_next()

    return images, {'labels': labels, 'mask': mask}, it

  return _input_fn


def get_local_dataset(is_training):
  '''
  :param is_training:
  :return: Read the raw data into a dataset object
  '''
  if FLAGS.train_mode == 'pretrain':
    if is_training:
      path = FLAGS.data_dir + '/unlabeled/train'
    else:
      path = FLAGS.data_dir + '/unlabeled/test'

    files = list(pathlib.Path(path).glob('*.jpg'))
    files = [str(f) for f in files]
    dataset = tf1.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files))

    def preprocess_img(filename):
      img = tf1.io.read_file(filename)
      img = tf1.io.decode_jpeg(img)
      return img, tf1.constant(0, dtype=tf1.int32)

    dataset = dataset.map(preprocess_img, num_parallel_calls=tf1.data.experimental.AUTOTUNE)
    return dataset

  elif FLAGS.train_mode == 'finetune':
    if is_training:
      path = FLAGS.data_dir + '/labeled/train'
    else: # evaluation
      if FLAGS.mode == 'train_then_eval':  # using validation set
        path = FLAGS.data_dir + '/labeled/val'
      else:  # using test set
        path = FLAGS.data_dir + '/labeled/test'

    files = list(pathlib.Path(path).glob('**/*.jpg'))
    files = [(str(f), f.parent.name) for f in files]
    dataset = tf1.data.Dataset.from_tensor_slices(files)
    dataset = dataset.shuffle(len(files))

    def preprocess_img(file_item):
      img = tf1.io.read_file(file_item[0])
      img = tf1.io.decode_jpeg(img)
      return img, tf1.strings.to_number(file_item[1], out_type=tf1.dtypes.float32)

    dataset = dataset.map(preprocess_img, num_parallel_calls=tf1.data.experimental.AUTOTUNE)

    return dataset


def pad_to_batch(dataset, batch_size):
  """Pad Tensors to specified batch size.

  Args:
    dataset: An instance of tf.data.Dataset.
    batch_size: The number of samples per batch of input requested.

  Returns:
    An instance of tf.data.Dataset that yields the same Tensors with the same
    structure as the original padded to batch_size along the leading
    dimension.

  Raises:
    ValueError: If the dataset does not comprise any tensors; if a tensor
      yielded by the dataset has an unknown number of dimensions or is a
      scalar; or if it can be statically determined that tensors comprising
      a single dataset element will have different leading dimensions.
  """

  def _pad_to_batch(*args):
    """Given Tensors yielded by a Dataset, pads all to the batch size."""
    flat_args = tf1.nest.flatten(args)

    for tensor in flat_args:
      if tensor.shape.ndims is None:
        raise ValueError(
          'Unknown number of dimensions for tensor %s.' % tensor.name)
      if tensor.shape.ndims == 0:
        raise ValueError('Tensor %s is a scalar.' % tensor.name)

    # This will throw if flat_args is empty. However, as of this writing,
    # tf.data.Dataset.map will throw first with an internal error, so we do
    # not check this case explicitly.
    first_tensor = flat_args[0]
    first_tensor_shape = tf1.shape(first_tensor)
    first_tensor_batch_size = first_tensor_shape[0]
    difference = batch_size - first_tensor_batch_size

    for i, tensor in enumerate(flat_args):
      control_deps = []
      if i != 0:
        # Check that leading dimensions of this tensor matches the first,
        # either statically or dynamically. (If the first dimensions of both
        # tensors are statically known, the we have to check the static
        # shapes at graph construction time or else we will never get to the
        # dynamic assertion.)
        if (first_tensor.shape[:1].is_fully_defined() and
                tensor.shape[:1].is_fully_defined()):
          if first_tensor.shape[0] != tensor.shape[0]:
            raise ValueError(
              'Batch size of dataset tensors does not match. %s '
              'has shape %s, but %s has shape %s' % (
                first_tensor.name, first_tensor.shape,
                tensor.name, tensor.shape))
        else:
          curr_shape = tf1.shape(tensor)
          control_deps = [tf1.Assert(
            tf1.equal(curr_shape[0], first_tensor_batch_size),
            ['Batch size of dataset tensors %s and %s do not match. '
             'Shapes are' % (tensor.name, first_tensor.name), curr_shape,
             first_tensor_shape])]

      with tf1.control_dependencies(control_deps):
        # Pad to batch_size along leading dimension.
        flat_args[i] = tf1.pad(
          tensor, [[0, difference]] + [[0, 0]] * (tensor.shape.ndims - 1))
      flat_args[i].set_shape([batch_size] + tensor.shape.as_list()[1:])

    return tf1.nest.pack_sequence_as(args, flat_args)

  return dataset.map(_pad_to_batch)
