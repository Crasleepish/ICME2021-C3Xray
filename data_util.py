"""Data preprocessing and augmentation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import flags

import tensorflow.compat.v1 as tf1
import tensorflow_addons as tfa
import math

FLAGS = flags.FLAGS

CROP_PROPORTION = 0.875  # Standard for ImageNet.
PI = 3.141592653589793


def preprocess_image(image, op_center, height, width, is_training=False,
                     color_distort=True, crop=True, flip=True, test_crop=True, rotate=False,
                     ):
  """Preprocesses the given image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    is_training: `bool` for whether the preprocessing is for training.
    color_distort: whether to apply the color distortion.
    crop: whether to use crop
    flip: whether to use flip
    test_crop: whether or not to extract a central crop of the images
        (as for standard ImageNet evaluation) during the evaluation.
    rotate: whether to use rotate

  Returns:
    A preprocessed image `Tensor` of range [0, 1].
  """
  image = tf1.image.convert_image_dtype(image, dtype=tf1.float32)
  if is_training:
    return preprocess_for_train(image, op_center, height, width, color_distort, crop, flip, rotate)
  else:
    return preprocess_for_eval(image, height, width, test_crop)


def preprocess_for_train(image, op_center, height, width,
                         color_distort=True, crop=True, flip=True, rotate=True):
  """Preprocesses the given image for training.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    color_distort: Whether to apply the color distortion.
    crop: Whether to crop the image.
    flip: Whether or not to flip left and right of an image.
    rotate: whether to use rotate

  Returns:
    A preprocessed image `Tensor`.
  """
  angle = 0.
  ori_image = image
  if rotate:
    image, angle = random_rotate_by_center(image, height, width, op_center)
  if crop:
    image = random_crop_with_resize(image, op_center, height, width, angle)
    image = tf1.cond(tf1.reduce_all(tf1.equal(image, tf1.zeros_like(image))),
                    lambda: random_crop_with_resize(image, op_center, height, width, 0.),
                    lambda: image)
    image = tf1.cond(tf1.reduce_all(tf1.equal(image, tf1.zeros_like(image))),
                    lambda: ori_image,
                    lambda: image)
  if flip:
    image = tf1.image.random_flip_left_right(image)
  if color_distort:
    image = random_color_jitter(image)
  image = tf1.reshape(image, [height, width, 3])
  image = tf1.clip_by_value(image, 0., 1.)
  return image


def random_color_jitter(image, p=1.0):
  def _transform(image):
    color_jitter_t = functools.partial(
        color_jitter, strength=FLAGS.color_jitter_strength)
    image = random_apply(color_jitter_t, p=0.8, x=image)
    return random_apply(to_grayscale, p=0.2, x=image)
  return random_apply(_transform, p=p, x=image)


def to_grayscale(image, keep_channels=True):
  image = tf1.image.rgb_to_grayscale(image)
  if keep_channels:
    image = tf1.tile(image, [1, 1, 3])
  return image


def color_jitter(image,
                 strength,
                 random_order=True):
  """Distorts the color of the image.

  Args:
    image: The input image tensor.
    strength: the floating number for the strength of the color augmentation.
    random_order: A bool, specifying whether to randomize the jittering order.

  Returns:
    The distorted image tensor.
  """
  brightness = 0.8 * strength
  contrast = 0.8 * strength
  saturation = 0.8 * strength
  hue = 0.2 * strength
  if random_order:
    return color_jitter_rand(image, brightness, contrast, saturation, hue)
  else:
    return color_jitter_nonrand(image, brightness, contrast, saturation, hue)


def color_jitter_nonrand(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image (jittering order is fixed).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf1.name_scope('distort_color'):
    def apply_transform(i, x, brightness, contrast, saturation, hue):
      """Apply the i-th transformation."""
      if brightness != 0 and i == 0:
        x = random_brightness(x, max_delta=brightness)
      elif contrast != 0 and i == 1:
        x = tf1.image.random_contrast(
            x, lower=1-contrast, upper=1+contrast)
      elif saturation != 0 and i == 2:
        x = tf1.image.random_saturation(
            x, lower=1-saturation, upper=1+saturation)
      elif hue != 0:
        x = tf1.image.random_hue(x, max_delta=hue)
      return x

    for i in range(4):
      image = apply_transform(i, image, brightness, contrast, saturation, hue)
      image = tf1.clip_by_value(image, 0., 1.)
    return image


def color_jitter_rand(image, brightness=0, contrast=0, saturation=0, hue=0):
  """Distorts the color of the image (jittering order is random).

  Args:
    image: The input image tensor.
    brightness: A float, specifying the brightness for color jitter.
    contrast: A float, specifying the contrast for color jitter.
    saturation: A float, specifying the saturation for color jitter.
    hue: A float, specifying the hue for color jitter.

  Returns:
    The distorted image tensor.
  """
  with tf1.name_scope('distort_color'):
    def apply_transform(i, x):
      """Apply the i-th transformation."""
      def brightness_foo():
        if brightness == 0:
          return x
        else:
          return random_brightness(x, max_delta=brightness)
      def contrast_foo():
        if contrast == 0:
          return x
        else:
          return tf1.image.random_contrast(x, lower=1-contrast, upper=1+contrast)
      def saturation_foo():
        if saturation == 0:
          return x
        else:
          return tf1.image.random_saturation(
              x, lower=1-saturation, upper=1+saturation)
      def hue_foo():
        if hue == 0:
          return x
        else:
          return tf1.image.random_hue(x, max_delta=hue)
      x = tf1.cond(tf1.less(i, 2),
                  lambda: tf1.cond(tf1.less(i, 1), brightness_foo, contrast_foo),
                  lambda: tf1.cond(tf1.less(i, 3), saturation_foo, hue_foo))
      return x

    perm = tf1.random_shuffle(tf1.range(4))
    for i in range(4):
      image = apply_transform(perm[i], image)
      image = tf1.clip_by_value(image, 0., 1.)
    return image


def random_brightness(image, max_delta, impl='simclrv2'):
  """A multiplicative vs additive change of brightness."""
  if impl == 'simclrv2':
    factor = tf1.random_uniform(
        [], tf1.maximum(1.0 - max_delta, 0), 1.0 + max_delta)
    image = image * factor
  elif impl == 'simclrv1':
    image = random_brightness(image, max_delta=max_delta)
  else:
    raise ValueError('Unknown impl {} for random brightness.'.format(impl))
  return image


def preprocess_for_eval(image, height, width, crop=True):
  """Preprocesses the given image for evaluation.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    crop: Whether or not to (center) crop the test images.

  Returns:
    A preprocessed image `Tensor`.
  """
  if crop:
    image = center_crop(image, height, width, crop_proportion=CROP_PROPORTION)
  image = tf1.reshape(image, [height, width, 3])
  image = tf1.clip_by_value(image, 0., 1.)
  return image


def _compute_crop_shape(
    image_height, image_width, aspect_ratio, crop_proportion):
  """Compute aspect ratio-preserving shape for central crop.

  The resulting shape retains `crop_proportion` along one side and a proportion
  less than or equal to `crop_proportion` along the other side.

  Args:
    image_height: Height of image to be cropped.
    image_width: Width of image to be cropped.
    aspect_ratio: Desired aspect ratio (width / height) of output.
    crop_proportion: Proportion of image to retain along the less-cropped side.

  Returns:
    crop_height: Height of image after cropping.
    crop_width: Width of image after cropping.
  """
  image_width_float = tf1.cast(image_width, tf1.float32)
  image_height_float = tf1.cast(image_height, tf1.float32)

  def _requested_aspect_ratio_wider_than_image():
    crop_height = tf1.cast(tf1.rint(
        crop_proportion / aspect_ratio * image_width_float), tf1.int32)
    crop_width = tf1.cast(tf1.rint(
        crop_proportion * image_width_float), tf1.int32)
    return crop_height, crop_width

  def _image_wider_than_requested_aspect_ratio():
    crop_height = tf1.cast(
        tf1.rint(crop_proportion * image_height_float), tf1.int32)
    crop_width = tf1.cast(tf1.rint(
        crop_proportion * aspect_ratio *
        image_height_float), tf1.int32)
    return crop_height, crop_width

  return tf1.cond(
      aspect_ratio > image_width_float / image_height_float,
      _requested_aspect_ratio_wider_than_image,
      _image_wider_than_requested_aspect_ratio)


def random_blur(image, height, width, p=1.0):
  """Randomly blur an image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    p: probability of applying this transformation.

  Returns:
    A preprocessed image `Tensor`.
  """
  del width
  def _transform(image):
    sigma = tf1.random.uniform([], 0.1, 2.0, dtype=tf1.float32)
    return gaussian_blur(
        image, kernel_size=height//10, sigma=sigma, padding='SAME')
  return random_apply(_transform, p=p, x=image)


def batch_random_blur(images_list, height, width, blur_probability=0.5):
  """Apply efficient batch data transformations.

  Args:
    images_list: a list of image tensors.
    height: the height of image.
    width: the width of image.
    blur_probability: the probaility to apply the blur operator.

  Returns:
    Preprocessed feature list.
  """
  def generate_selector(p, bsz):
    shape = [bsz, 1, 1, 1]
    selector = tf1.cast(
        tf1.less(tf1.random_uniform(shape, 0, 1, dtype=tf1.float32), p),
        tf1.float32)
    return selector

  new_images_list = []
  for images in images_list:
    images_new = random_blur(images, height, width, p=1.)
    selector = generate_selector(blur_probability, tf1.shape(images)[0])
    images = images_new * selector + images * (1 - selector)
    images = tf1.clip_by_value(images, 0., 1.)
    new_images_list.append(images)

  return new_images_list


def gaussian_blur(image, kernel_size, sigma, padding='SAME'):
  """Blurs the given image with separable convolution.


  Args:
    image: Tensor of shape [height, width, channels] and dtype float to blur.
    kernel_size: Integer Tensor for the size of the blur kernel. This is should
      be an odd number. If it is an even number, the actual kernel size will be
      size + 1.
    sigma: Sigma value for gaussian operator.
    padding: Padding to use for the convolution. Typically 'SAME' or 'VALID'.

  Returns:
    A Tensor representing the blurred image.
  """
  radius = tf1.to_int32(kernel_size / 2)
  kernel_size = radius * 2 + 1
  x = tf1.to_float(tf1.range(-radius, radius + 1))
  blur_filter = tf1.exp(
      -tf1.pow(x, 2.0) / (2.0 * tf1.pow(tf1.to_float(sigma), 2.0)))
  blur_filter /= tf1.reduce_sum(blur_filter)
  # One vertical and one horizontal filter.
  blur_v = tf1.reshape(blur_filter, [kernel_size, 1, 1, 1])
  blur_h = tf1.reshape(blur_filter, [1, kernel_size, 1, 1])
  num_channels = tf1.shape(image)[-1]
  blur_h = tf1.tile(blur_h, [1, 1, num_channels, 1])
  blur_v = tf1.tile(blur_v, [1, 1, num_channels, 1])
  expand_batch_dim = image.shape.ndims == 3
  if expand_batch_dim:
    # Tensorflow requires batched input to convolutions, which we can fake with
    # an extra dimension.
    image = tf1.expand_dims(image, axis=0)
  blurred = tf1.nn.depthwise_conv2d(
      image, blur_h, strides=[1, 1, 1, 1], padding=padding)
  blurred = tf1.nn.depthwise_conv2d(
      blurred, blur_v, strides=[1, 1, 1, 1], padding=padding)
  if expand_batch_dim:
    blurred = tf1.squeeze(blurred, axis=0)
  return blurred


def center_crop(image, height, width, crop_proportion):
  """Crops to center of image and rescales to desired size.

  Args:
    image: Image Tensor to crop.
    height: Height of image to be cropped.
    width: Width of image to be cropped.
    crop_proportion: Proportion of image to retain along the less-cropped side.

  Returns:
    A `height` x `width` x channels Tensor holding a central crop of `image`.
  """
  shape = tf1.shape(image)
  image_height = shape[0]
  image_width = shape[1]
  crop_height, crop_width = _compute_crop_shape(
      image_height, image_width, height / width, crop_proportion)
  offset_height = ((image_height - crop_height) + 1) // 2
  offset_width = ((image_width - crop_width) + 1) // 2
  image = tf1.image.crop_to_bounding_box(
      image, offset_height, offset_width, crop_height, crop_width)

  image = tf1.image.resize_bicubic([image], [height, width])[0]

  return image


def random_rotate_by_center(image, height, width, center, p=1.):
  '''
  Rotate the given image by a random angle(in radians) around a specific center.
  :param
  image: The input image(HWC)
  height: The height of the image, int32
  width: The width of the image, int32
  center: The center point of rotation, a 2-dim vector in range [0, 1]
  p: Probability of rotation
  :return:
  image: The rotated image
  angle: The sample angle
  '''
  angle_range_factor = FLAGS.rotate_angle_range_factor
  prob_mask = tf1.math.floor(tf1.random.uniform([], 0., 1.) + p)
  angle = tf1.random.uniform([], -PI / angle_range_factor, PI / angle_range_factor, dtype=tf1.float32)
  angle = angle * prob_mask
  image = tf1.pad(image, [[height//2, height//2], [width//2, width//2], [0, 0]])
  v = tf1.constant([width/2.0, height/2.0]) - center * tf1.constant([width*1.0, height*1.0])
  image = tfa.image.translate(image, v)
  image = tfa.image.rotate(image, angle, interpolation='BILINEAR')
  image = tfa.image.translate(image, -v)
  image = image[height // 2:height * 3 // 2, width // 2:width * 3 // 2, :]
  return image, angle


def random_crop_with_resize(image, op_center, height, width, angle=0., p=1.0):
  """Randomly crop and resize an image.

  Args:
    image: `Tensor` representing an image of arbitrary size.
    height: Height of output image.
    width: Width of output image.
    angle: angle of rotated when rotation has been used, different angle
      may influence the choosing of bounding box for cropping
    crop_center_id: Which center point has been choosen for random crop
    p: Probability of applying this transformation.

  Returns:
    A preprocessed image `Tensor`.
  """
  def _transform(image):  # pylint: disable=missing-docstring
    image = crop_and_resize(image, op_center, height, width, angle)
    return image
  return random_apply(_transform, p=p, x=image)


def random_apply(func, p, x):
  """Randomly apply function func to x with probability p."""
  return tf1.cond(
      tf1.less(tf1.random_uniform([], minval=0, maxval=1, dtype=tf1.float32),
              tf1.cast(p, tf1.float32)),
      lambda: func(x),
      lambda: x)


def crop_and_resize(image, op_center, height, width, angle=0.):
  """Make a random crop and resize it to height `height` and width `width`.

  Args:
    image: Tensor representing the image.
    height: Desired image height.
    width: Desired image width.
    angle: angle of rotated when rotation has been used, different angle
      may influence the choosing of bounding box for cropping

  Returns:
    A `height` x `width` x channels Tensor holding a random crop of `image`.
  """
  bbox = tf1.constant([0.0, 0.0, 1.0, 1.0], dtype=tf1.float32, shape=[4])
  aspect_ratio = width / height
  image = distorted_bounding_box_crop(
      image,
      op_center,
      bbox,
      min_object_covered=0.1,
      aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio),
      area_range=(FLAGS.crop_area_lower_bound, 1.0),  # lower bound of area_range should not be larger than 0.72
      max_attempts=200,
      angle=angle,
      scope=None)
  return tf1.image.resize_bicubic([image], [height, width])[0]


def distorted_bounding_box_crop(image,
                                op_center,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                angle=0.,
                                scope=None):
  """Generates cropped_image using one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: `Tensor` of image data.
    bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
        where each coordinate is [0, 1) and the coordinates are arranged
        as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
        image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
        area of the image must contain at least this fraction of any bounding
        box supplied.
    aspect_ratio_range: An optional list of `float`s. The cropped area of the
        image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `float`s. The cropped area of the image
        must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
        region of the image of the specified constraints. After `max_attempts`
        failures, return the entire image.
    angle: angle of rotated when rotation has been used, different angle
        may influence the choosing of bounding box for cropping
    scope: Optional `str` for name scope.
  Returns:
    (cropped image `Tensor`, distorted bbox `Tensor`).
  """
  with tf1.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    shape = tf1.shape(image)
    sample_distorted_bounding_box = random_sample_bounding_box(
        shape,
        op_center,
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True,
        sample_angle=angle
    )
    bbox_begin, bbox_size = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    offset_y, offset_x = tf1.unstack(bbox_begin)
    target_height, target_width = tf1.unstack(bbox_size)
    image = tf1.cond(tf1.equal(target_height, -1),
                    lambda: tf1.zeros_like(image),
                    lambda: tf1.image.crop_to_bounding_box(image, offset_y, offset_x, target_height, target_width)
                    )

    return image


def random_sample_bounding_box(shape, op_center, bounding_boxes, min_object_covered, aspect_ratio_range,
                               area_range, max_attempts, use_image_if_no_bounding_boxes, sample_angle):
  with tf1.name_scope("part1"):
    width = tf1.cast(shape[0], 'float32')
    height = tf1.cast(shape[1], 'float32')
    range_x_1 = width * bounding_boxes[0]
    range_y_1 = height * bounding_boxes[1]
    range_x_2 = width * bounding_boxes[2]
    range_y_2 = height * bounding_boxes[3]

  with tf1.name_scope("part2"):
    if FLAGS.crop_strategy == 'centerized':
      rand_point_1 = [tf1.random.uniform([max_attempts], 0.0, 0.25*width, dtype=tf1.float32),
                      tf1.random.uniform([max_attempts], 0.0, 0.25*height, dtype=tf1.float32)]
      rand_point_2 = [tf1.random.uniform([max_attempts], 0.75*width, 1.0*width, dtype=tf1.float32),
                      tf1.random.uniform([max_attempts], 0.75*height, 1.0*height, dtype=tf1.float32)]
      r_centers = tf1.stack([width*0.5, height*0.5])
    elif FLAGS.crop_strategy == 'center_specific':
      top_left_bound_min = tf1.constant([0.0, 0.0], dtype='float32')
      top_left_bound_max = op_center * tf1.stack([width*1.0, height*1.0])
      bottom_right_bound_min = op_center * tf1.stack([width*1.0, height*1.0])
      bottom_right_bound_max = tf1.stack([width*1.0, height*1.0])

      rand_point_1 = [tf1.random.uniform([max_attempts], top_left_bound_min[0], top_left_bound_max[0], dtype=tf1.float32),
                      tf1.random.uniform([max_attempts], top_left_bound_min[1], top_left_bound_max[1], dtype=tf1.float32)]
      rand_point_2 = [tf1.random.uniform([max_attempts], bottom_right_bound_min[0], bottom_right_bound_max[0], dtype=tf1.float32),
                      tf1.random.uniform([max_attempts], bottom_right_bound_min[1], bottom_right_bound_max[1], dtype=tf1.float32)]
      r_centers = op_center * tf1.stack([width*1.0, height*1.0])
    else:
      rand_point_1 = [tf1.random.uniform([max_attempts], range_x_1, range_x_2, dtype=tf1.float32),
                      tf1.random.uniform([max_attempts], range_y_1, range_y_2, dtype=tf1.float32)]
      rand_point_2 = [tf1.random.uniform([max_attempts], range_x_1, range_x_2, dtype=tf1.float32),
                      tf1.random.uniform([max_attempts], range_y_1, range_y_2, dtype=tf1.float32)]
      r_centers = tf1.stack([width*0.5, height*0.5])
    left_up_points = [tf1.math.round(tf1.minimum(rand_point_1[0], rand_point_2[0])),
                      tf1.math.round(tf1.minimum(rand_point_1[1], rand_point_2[1]))]
    sizes = [tf1.math.round(tf1.abs(rand_point_1[0] - rand_point_2[0])),
             tf1.math.round(tf1.abs(rand_point_1[1] - rand_point_2[1]))]

  with tf1.name_scope("part3"):
    if use_image_if_no_bounding_boxes:
      # add -1 flags, when all attemps fails, -1 flags should be chosen
      left_up_points[0] = tf1.concat([left_up_points[0], [-1.]], axis=-1)
      left_up_points[1] = tf1.concat([left_up_points[1], [-1.]], axis=-1)
      sizes[0] = tf1.concat([sizes[0], [-1.]], axis=-1)
      sizes[1] = tf1.concat([sizes[1], [-1.]], axis=-1)

    right_up_points = [left_up_points[0] + sizes[0],
                       left_up_points[1]]
    left_bottom_points = [left_up_points[0], left_up_points[1] + sizes[1]]
    right_bottom_points = [left_up_points[0] + sizes[0],
                           left_up_points[1] + sizes[1]]

  def rotate_clockwise(x, y, center_x, center_y, sample_angle):
    x = x - center_x
    y = y - center_y
    x_prime = x * tf1.cos(sample_angle) - y * tf1.sin(sample_angle)
    y_prime = x * tf1.sin(sample_angle) + y * tf1.cos(sample_angle)
    x = center_x + x_prime
    y = center_y + y_prime
    return x, y

  with tf1.name_scope("part4"):
    rotate_center = [r_centers[0], r_centers[1]]
    # left_up_points should be valid
    left_up_points_clock_rotated = rotate_clockwise(left_up_points[0],
                                                    left_up_points[1],
                                                    rotate_center[0], rotate_center[1], sample_angle)
    left_up_points_valid_cond = tf1.logical_and(
      tf1.logical_and(tf1.greater(left_up_points_clock_rotated[0], range_x_1),
                     tf1.less(left_up_points_clock_rotated[0], range_x_2)),
      tf1.logical_and(tf1.greater(left_up_points_clock_rotated[1], range_y_1),
                     tf1.less(left_up_points_clock_rotated[1], range_y_2))
    )

    # right up points should be valid
    right_up_points_clock_rotated = rotate_clockwise(right_up_points[0],
                                                     right_up_points[1],
                                                     rotate_center[0], rotate_center[1], sample_angle)
    right_up_points_valid_cond = tf1.logical_and(
      tf1.logical_and(tf1.greater(right_up_points_clock_rotated[0], range_x_1),
                     tf1.less(right_up_points_clock_rotated[0], range_x_2)),
      tf1.logical_and(tf1.greater(right_up_points_clock_rotated[1], range_y_1),
                     tf1.less(right_up_points_clock_rotated[1], range_y_2))
    )

    # left_bottom_points_should be valid
    left_bottom_points_clock_rotated = rotate_clockwise(left_bottom_points[0],
                                                        left_bottom_points[1],
                                                        rotate_center[0], rotate_center[1], sample_angle)
    left_bottom_points_valid_cond = tf1.logical_and(
      tf1.logical_and(tf1.greater(left_bottom_points_clock_rotated[0], range_x_1),
                     tf1.less(left_bottom_points_clock_rotated[0], range_x_2)),
      tf1.logical_and(tf1.greater(left_bottom_points_clock_rotated[1], range_y_1),
                     tf1.less(left_bottom_points_clock_rotated[1], range_y_2))
    )

    # right bottom points should be valid
    right_bottom_points_clock_rotated = rotate_clockwise(right_bottom_points[0],
                                                         right_bottom_points[1],
                                                         rotate_center[0], rotate_center[1], sample_angle)
    right_bottom_points_valid_cond = tf1.logical_and(
      tf1.logical_and(tf1.greater(right_bottom_points_clock_rotated[0], range_x_1),
                     tf1.less(right_bottom_points_clock_rotated[0], range_x_2)),
      tf1.logical_and(tf1.greater(right_bottom_points_clock_rotated[1], range_y_1),
                     tf1.less(right_bottom_points_clock_rotated[1], range_y_2))
    )

  with tf1.name_scope("part5"):
    # area should be valid
    min_object_covered_cond = tf1.greater(sizes[0] * sizes[1],
      min_object_covered * tf1.abs(range_x_1 - range_x_2) * tf1.abs(range_y_1 - range_y_2)
    )

    area_range_cond = tf1.logical_and(
      tf1.greater(sizes[0] * sizes[1], area_range[0] * width * height),
      tf1.less(sizes[0] * sizes[1], area_range[1] * width * height)
    )

    # aspect ratio should be valid
    aspect_ratio_cond = tf1.logical_and(
      tf1.greater(sizes[0] / sizes[1], aspect_ratio_range[0]),
      tf1.less(sizes[0] / sizes[1], aspect_ratio_range[1])
    )

  with tf1.name_scope("part6"):
    cond = tf1.cast(tf1.ones_like(left_up_points[0]), dtype='bool')
    cond = tf1.logical_and(cond, left_up_points_valid_cond)
    cond = tf1.logical_and(cond, right_up_points_valid_cond)
    cond = tf1.logical_and(cond, left_bottom_points_valid_cond)
    cond = tf1.logical_and(cond, right_bottom_points_valid_cond)
    cond = tf1.logical_and(cond, min_object_covered_cond)
    cond = tf1.logical_and(cond, area_range_cond)
    cond = tf1.logical_and(cond, aspect_ratio_cond)

    find_idx = tf1.squeeze(tf1.where(cond), axis=1)
    find_idx = tf1.concat([find_idx, [-1]], axis=0)
    find_idx = find_idx[0]

  with tf1.name_scope("part7"):
    bbox_begin = tf1.stack([tf1.cast(left_up_points[1][find_idx], dtype='int32'),
                           tf1.cast(left_up_points[0][find_idx], dtype='int32')])
    bbox_size = tf1.stack([tf1.cast(sizes[1][find_idx], dtype='int32'),
                          tf1.cast(sizes[0][find_idx], dtype='int32')])
  return bbox_begin, bbox_size
