# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tensorflow as tf

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94
_CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]

INPUT_SIZE = 224
CROP_PADDING = 8

def mean_image_subtraction(image, means, num_channels):
  if image.get_shape().ndims != 3:
      raise ValueError('Input must be of size [height, width, C>0]')
  if len(means) != num_channels:
      raise ValueError('len(means) must match the number of channels')
  
  means = tf.expand_dims(tf.expand_dims(means, 0), 0)
  return image - means

def decode_and_center_crop(image_bytes, image_size=INPUT_SIZE):
  """Crops to center of image with padding then scales image_size."""
  shape = tf.image.extract_jpeg_shape(image_bytes)
  image_height = shape[0]
  image_width = shape[1]

  padded_center_crop_size = tf.cast(
      ((image_size / (image_size + CROP_PADDING)) *
        tf.cast(tf.minimum(image_height, image_width), tf.float32)),
        tf.int32)

  offset_height = ((image_height - padded_center_crop_size) + 1) // 2
  offset_width = ((image_width - padded_center_crop_size) + 1) // 2
  crop_window = tf.stack([offset_height, offset_width,
                          padded_center_crop_size, padded_center_crop_size])
  image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
  image = tf.image.resize_bilinear([image], [image_size, image_size])[0]

  return image

def preprocess_image(image_buffer,
                     output_height,
                     output_width,
                     num_channels=3
                     ):
  # For validation, we want to decode, resize, then just crop the middle.
  image = decode_and_center_crop(image_buffer)
  return mean_image_subtraction(image, _CHANNEL_MEANS, num_channels)
