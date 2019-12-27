# Lint as: python2, python3
# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Tests for tfx.tools.cli.handler.template_handler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow.compat.v2 as tf

from tfx.tools.cli import labels
from tfx.tools.cli.handler.template_handler import TemplateHandler


class TemplateHandlerTest(tf.test.TestCase):

  def setUp(self):
    super(TemplateHandlerTest, self).setUp()
    self.handler = TemplateHandler()

  def testList(self):
    templates = self.handler.list()
    self.assertNotEqual(templates, [])
    self.assertIn('classification', templates)

  def testCopy(self):
    test_dir = self.create_tempdir().full_path
    flags = {
        labels.MODEL: 'classification',
        labels.DESTINATION_PATH: test_dir,
        labels.PIPELINE_NAME: 'my_pipeline'
    }
    self.handler.copy(flags)
    copied_files = os.listdir(test_dir)
    self.assertNotEqual(copied_files, [])
    self.assertContainsSubset(['README', '__init__.py'], copied_files)


if __name__ == '__main__':
  tf.test.main()
