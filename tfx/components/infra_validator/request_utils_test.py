# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for tfx.utils.request_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf

from tensorflow_serving.apis import classification_pb2
from tensorflow_serving.apis import regression_pb2
from tfx.components.infra_validator import request_utils
from tfx.types import artifact_utils
from tfx.types import standard_artifacts


class RequestBuilderTest(tf.test.TestCase):

  def setUp(self):
    super(RequestBuilderTest, self).setUp()
    self._examples = standard_artifacts.Examples()
    self._examples.split_names = artifact_utils.encode_split_names(
        ['train', 'eval'])
    self._examples.uri = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'testdata',
        'csv_example_gen'
    )

  def testReadFromExamplesArtifact_UptoMaxExamples(self):
    builder = request_utils.RequestBuilder(max_examples=10, model_name='foo')

    builder.ReadFromExamplesArtifact(self._examples, split_name='eval')

    self.assertEqual(builder.num_examples, 10)

  def testReadFromExamplesArtifact_FailsIfSplitNotExists(self):
    builder = request_utils.RequestBuilder(max_examples=10, model_name='foo')

    with self.assertRaisesRegexp(ValueError, 'No split name asdf;'):
      builder.ReadFromExamplesArtifact(self._examples, split_name='asdf')

  def testBuildClassificationRequest(self):
    builder = request_utils.RequestBuilder(max_examples=2, model_name='foo')
    builder.ReadFromExamplesArtifact(self._examples, split_name='eval')

    requests = builder.BuildClassificationRequests()

    self.assertEqual(len(requests), 2)
    for request in requests:
      self.assertIsInstance(request, classification_pb2.ClassificationRequest)
      self.assertEqual(request.model_spec.name, 'foo')
      self.assertEqual(request.model_spec.signature_name, '')
      for example in request.input.example_list.examples:
        self.assertValidTaxiExample(example)

  def testBuildRegressionRequest(self):
    builder = request_utils.RequestBuilder(max_examples=2, model_name='foo')
    builder.ReadFromExamplesArtifact(self._examples, split_name='eval')

    requests = builder.BuildRegressionRequests()

    self.assertEqual(len(requests), 2)
    for request in requests:
      self.assertIsInstance(request, regression_pb2.RegressionRequest)
      self.assertEqual(request.model_spec.name, 'foo')
      self.assertEqual(request.model_spec.signature_name, '')
      for example in request.input.example_list.examples:
        self.assertValidTaxiExample(example)

  def assertValidTaxiExample(self, tf_example: tf.train.Example):
    features = tf_example.features.feature
    self.assertIntFeature(features['trip_start_day'])
    self.assertIntFeature(features['pickup_community_area'])
    self.assertStringFeature(features['payment_type'])
    self.assertFloatFeature(features['trip_miles'])
    self.assertIntFeature(features['trip_start_timestamp'])
    self.assertFloatFeature(features['pickup_latitude'])
    self.assertFloatFeature(features['pickup_longitude'])
    self.assertIntFeature(features['trip_start_month'])
    self.assertIntFeature(features['trip_start_hour'])
    self.assertFloatFeature(features['trip_seconds'])

  def assertFloatFeature(self, feature: tf.train.Feature):
    self.assertEqual(len(feature.float_list.value), 1)

  def assertIntFeature(self, feature: tf.train.Feature):
    self.assertEqual(len(feature.int64_list.value), 1)

  def assertStringFeature(self, feature: tf.train.Feature):
    self.assertEqual(len(feature.bytes_list.value), 1)


if __name__ == '__main__':
  tf.test.main()
