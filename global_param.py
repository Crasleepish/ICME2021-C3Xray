from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags

FLAGS = flags.FLAGS

flags.DEFINE_float(
    'learning_rate', 1e-4,
    'Initial learning rate per batch size of 256.')

flags.DEFINE_float(
    'warmup_epochs', 10,
    'Number of epochs of warmup.')

flags.DEFINE_float(
    'weight_decay', 1e-4,
    'Amount of weight decay to use.')

flags.DEFINE_float(
    'batch_norm_decay', 0.9,
    'Batch norm momentum.')

flags.DEFINE_integer(
    'train_batch_size', 512,
    'Batch size for training.')

flags.DEFINE_integer(
    'train_epochs', 100,
    'Number of epochs to train for.')

flags.DEFINE_integer(
    'train_steps', 0,
    'Number of steps to train for. If provided, overrides train_epochs.')

flags.DEFINE_integer(
    'eval_batch_size', 256,
    'Batch size for eval.')

flags.DEFINE_integer(
    'train_summary_steps', 100,
    'Steps before saving training summaries. If 0, will not save.')

flags.DEFINE_integer(
    'checkpoint_epochs', 1,
    'Number of epochs between checkpoints/summaries.')

flags.DEFINE_integer(
    'checkpoint_steps', 0,
    'Number of steps between checkpoints/summaries. If provided, overrides '
    'checkpoint_epochs.')

flags.DEFINE_enum(
    'mode', 'train', ['train', 'eval', 'train_then_eval'],
    'Whether to perform training or evaluation.')

flags.DEFINE_enum(
    'train_mode', 'pretrain', ['pretrain', 'finetune'],
    'The train mode controls different objectives and trainable components.')

flags.DEFINE_string(
    'checkpoint', None,
    'Loading from the given checkpoint for continued training or fine-tuning.')

flags.DEFINE_string(
    'variable_schema', '?!global_step',
    'This defines whether some variable from the checkpoint should be loaded.')

flags.DEFINE_bool(
    'zero_init_logits_layer', False,
    'If True, zero initialize layers after avg_pool for supervised learning.')

flags.DEFINE_integer(
    'fine_tune_after_block', -1,
    'The layers after which block that we will fine-tune. -1 means fine-tuning '
    'everything. 0 means fine-tuning after stem block. 4 means fine-tuning '
    'just the linear head.')

flags.DEFINE_string(
    'model_dir', None,
    'Model directory for training.')

flags.DEFINE_string(
    'data_dir', None,
    'Directory where dataset is stored.')

flags.DEFINE_float(
    'momentum', 0.9,
    'Momentum parameter.')

flags.DEFINE_string(
    'eval_name', None,
    'Name for eval.')

flags.DEFINE_integer(
    'keep_checkpoint_max', 5,
    'Maximum number of checkpoints to keep.')

flags.DEFINE_float(
    'temperature', 0.1,
    'Temperature parameter for contrastive loss.')

flags.DEFINE_boolean(
    'hidden_norm', True,
    'Whether or not to use normalization on the hidden vector.')

flags.DEFINE_enum(
    'proj_head_mode', 'nonlinear', ['none', 'linear', 'nonlinear'],
    'How the head projection is done.')

flags.DEFINE_integer(
    'proj_out_dim', 128,
    'Number of head projection dimension.')

flags.DEFINE_integer(
    'num_proj_layers', 3,
    'Number of non-linear head layers.')

flags.DEFINE_integer(
    'ft_proj_selector', 0,
    'Which layer of the projection head to use during fine-tuning. '
    '0 means throwing away the projection head, and -1 means the final layer.')

flags.DEFINE_boolean(
    'global_bn', False,
    'Whether to aggregate BN statistics across distributed cores.')

flags.DEFINE_integer(
    'width_multiplier', 1,
    'Multiplier to change width of network.')

flags.DEFINE_integer(
    'resnet_depth', 50,
    'Depth of ResNet.')

flags.DEFINE_float(
    'sk_ratio', 0.,
    'If it is bigger than 0, it will enable SK. Recommendation: 0.0625.')

flags.DEFINE_float(
    'se_ratio', 0.,
    'If it is bigger than 0, it will enable SE.')

flags.DEFINE_integer(
    'image_size', 224,
    'Input image size.')

flags.DEFINE_float(
    'color_jitter_strength', 1.0,
    'The strength of color jittering.')

flags.DEFINE_boolean(
    'use_blur', True,
    'Whether or not to use Gaussian blur for augmentation during pretraining.')

flags.DEFINE_boolean(
  'use_learning_rate_schedule', True,
  'Whether or not to use learning rate schedule.')

flags.DEFINE_boolean(
  'finetune_projection_head', True,
  'Whether or not to finetune projection head.')

flags.DEFINE_float(
    'rotate_angle_range_factor', 18.0,
    'The fraction of PI, indicating the angle of rotation augmentation. ')

flags.DEFINE_float(
    'crop_area_lower_bound', 0.1,
    'The ratio of how much area of the original image should be cropped out. '
    'This parameter should not be greater than 0.72, recommended to be 0.1 when pre-training,'
    '0.3 when direct-training, and 0.6 when attention-training.')

flags.DEFINE_float(
    'dropout_rate', 0.0,
    'The probability that each element is dropped.')

flags.DEFINE_integer(
    'num_classes', 1,
    'Number of classes. num_classes=1 indicates a binary classification task.')

flags.DEFINE_integer(
    'num_train_examples', 1024,
    'Number of training examples.')

flags.DEFINE_integer(
    'num_remain_examples', 1024,
    'Number of unlabeled examples for self-training.')

flags.DEFINE_integer(
    'num_eval_examples', 128,
    'Number of evaluation examples.')

flags.DEFINE_enum(
    'crop_strategy', 'center_specific', ['none', 'centerized', 'center_specific'],
    'The strategy used when applying crop augmentation. When using C3 strategy, '
    'we recommend choosing center_specific during pre-training and centerized'
    'during fine-tuning. ')

flags.DEFINE_integer(
    'half_life', 100,
    'Number of epochs required for learning rate to reduce to half.')

flags.DEFINE_boolean(
    'use_rotation', True,
    'Whether or not to use adjusted rotation augmentation.')

flags.DEFINE_boolean(
    'output_heatmap', True,
    'Whether or not to generate heatmap.'
)

