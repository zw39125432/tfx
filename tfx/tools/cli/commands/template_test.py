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
"""Tests for tfx.tools.cli.commands.copy_template."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import locale
import os

from click import testing as click_testing
import mock
import tensorflow.compat.v2 as tf

from tfx.tools.cli.commands import template
from tfx.tools.cli.commands.template import template_group


class TemplateTest(tf.test.TestCase):

  def setUp(self):
    super(TemplateTest, self).setUp()
    # Change the encoding for Click since Python 3 is configured to use ASCII as
    # encoding for the environment.
    if codecs.lookup(locale.getpreferredencoding()).name == 'ascii':
      os.environ['LANG'] = 'en_US.utf-8'
    self.runner = click_testing.CliRunner()
    self.addCleanup(mock.patch.stopall)
    mock.patch.object(template, 'TemplateHandler').start()

  def testListSuccess(self):
    result = self.runner.invoke(template_group, ['list'])
    self.assertEqual(0, result.exit_code)
    self.assertIn('Available templates', result.output)

  def testMissingPipelineName(self):
    result = self.runner.invoke(
        template_group, ['copy', '--model', 'm', '--destination_path', '/path'])
    self.assertNotEqual(0, result.exit_code)
    self.assertIn('pipeline_name', result.output)

  def testMissingDestDir(self):
    result = self.runner.invoke(
        template_group, ['copy', '--pipeline_name', 'p', '--model', 'm'])
    self.assertNotEqual(0, result.exit_code)
    self.assertIn('dest_dir', result.output)

  def testMissingModel(self):
    result = self.runner.invoke(
        template_group,
        ['copy', '--pipeline_name', 'p', '--destination_path', '/path'])
    self.assertNotEqual(0, result.exit_code)
    self.assertIn('model', result.output)

  def testCopySuccess(self):
    result = self.runner.invoke(template_group, [
        'copy', '--pipeline_name', 'p', '--destination_path', '/path',
        '--model', 'm'
    ])
    self.assertEqual(0, result.exit_code)
    self.assertIn('Copying', result.output)


if __name__ == '__main__':
  tf.test.main()
