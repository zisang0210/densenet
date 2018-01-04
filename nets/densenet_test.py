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
"""Tests for slim.denseNet."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from nets import densenet
slim = tf.contrib.slim

class DenseNetTest(tf.test.TestCase):

  def testAllEndPointsShapes(self):
    batch_size = 5
    height, width = 224, 224
    num_classes = 1000
    inputs = tf.random_uniform((batch_size, height, width, 3))
    with slim.arg_scope(densenet.densenet_arg_scope()):
        logits, end_points = densenet.densenet(inputs, num_classes)
    endpoints_shapes = {'input_layer': [batch_size, 56, 56, 48],
                        'dense_block1': [batch_size, 56, 56, 192],
                        'transition_layer1': [batch_size, 28, 28, 96],
                        'dense_block2': [batch_size, 28, 28, 384],
                        'transition_layer2': [batch_size, 14, 14, 192],
                        'dense_block3': [batch_size, 14, 14, 768],
                        'transition_layer3': [batch_size, 7, 7, 384],
                        'dense_block4': [batch_size, 7, 7, 768],
                        'classify_layer': [batch_size, num_classes]}
    self.assertItemsEqual(endpoints_shapes.keys(), end_points.keys())
    for endpoint_name in endpoints_shapes:
      expected_shape = endpoints_shapes[endpoint_name]
      self.assertTrue(endpoint_name in end_points)
      self.assertListEqual(end_points[endpoint_name].get_shape().as_list(),
                           expected_shape)


if __name__ == '__main__':
  tf.test.main()
