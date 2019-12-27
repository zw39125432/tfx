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
"""Handler for Template related operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Text, Dict, Any, List
import click

from tfx.tools.cli import labels

_PLACEHOLDER_PIPELINE_NAME = '{{PIPELINE_NAME}}'


class TemplateHandler(object):
  """Template Handler for CLI.

  Handles operations for templates in tfx/experimental/templates/ directory.
  """

  def _template_src_dir(self) -> Text:
    """Get template directory in the source tree.

      We should find tfx/experimental/templates
      from tfx/tools/cli/handler/template_handler.py.
    Returns:
      Path to the directory containing template sources.
    """
    return os.path.join(
        os.path.dirname(  # tfx/
            os.path.dirname(  # tools/
                os.path.dirname(  # cli/
                    os.path.dirname(os.path.abspath(__file__))))),  # handler/
        'experimental',
        'templates')

  def list(self) -> List[Text]:
    """List available templates by inspecting template source directory.

    Returns:
      List of template names which is same as directory name.
    """
    templates_dir = self._template_src_dir()
    names = []
    for f in os.listdir(templates_dir):
      if f.startswith('_'):
        continue
      if not os.path.isdir(os.path.join(templates_dir, f)):
        continue
      names.append(f)
    return names

  def _copy_and_replace_placeholder_dir(self, src: Text, dst: Text,
                                        replace_dict: Dict[Text, Text]) -> None:
    """Copy a directory to destination path and replace the placeholders."""
    if not os.path.isdir(dst):
      if os.path.exists(dst):
        raise RuntimeError(
            'Cannot copy template directory {}. Already a file exists.'.format(
                src))
      os.mkdir(dst)
    for f in os.listdir(src):
      src_file = os.path.join(src, f)
      dst_file = os.path.join(dst, f)
      if os.path.isdir(src_file):
        if f.startswith('_'):  # Excludes __pycache__ and other private folders.
          continue
        self._copy_and_replace_placeholder_dir(src_file, dst_file, replace_dict)
      else:  # a file.
        self._copy_and_replace_placeholder_file(src_file, dst_file,
                                                replace_dict)

  def _copy_and_replace_placeholder_file(
      self, src: Text, dst: Text, replace_dict: Dict[Text, Text]) -> None:
    """Copy a file to destination path and replace the placeholders."""
    click.echo('{} -> {}'.format(os.path.basename(src), dst))
    with open(src) as fp:
      contents = fp.read()
    for orig, new in replace_dict.items():
      contents = contents.replace(orig, new)
    with open(dst, 'w') as fp:
      fp.write(contents)

  def copy(self, flags_dict: Dict[Text, Any]) -> None:
    """Copy template flags_dict["model"] to flags_dict["dest_dir"].

    Copies all *.py and README files in specified template, and replace
    placeholder {{PIPELINE_NAME}} with specified name in
    flags_dict["pipeline_name"].

    Args:
      flags_dict: Should have pipeline_name, model and dest_dir.
    """
    pipeline_name = flags_dict[labels.PIPELINE_NAME]
    if '/' in pipeline_name:
      raise ValueError(
          'Invalid pipeline name "{}". "/" cannot be included.'.format(
              pipeline_name))

    template_dir = os.path.join(self._template_src_dir(),
                                flags_dict[labels.MODEL])
    if not os.path.isdir(template_dir):
      raise ValueError('Model {} does not exist.'.format(
          flags_dict[labels.MODEL]))

    replace_dict = {
        _PLACEHOLDER_PIPELINE_NAME: pipeline_name,
    }
    self._copy_and_replace_placeholder_dir(template_dir,
                                           flags_dict[labels.DESTINATION_PATH],
                                           replace_dict)
