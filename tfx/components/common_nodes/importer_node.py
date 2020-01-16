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
"""TFX Importer definition."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Type

import absl

from tfx import types
from tfx.components.base import base_driver
from tfx.components.base import base_node
from tfx.orchestration import data_types
from tfx.types import channel_utils
from tfx.types import node_common

# Constant to access importer importing result from importer output dict.
IMPORT_RESULT_KEY = 'result'
# Constant to access artifact uri from importer exec_properties dict.
SOURCE_URI_KEY = 'artifact_uri'
# Constant to access artifact properties from importer exec_properties dict.
PROPERTIES_KEY = 'artifact_properties'
# Constant to access re-import option from importer exec_properties dict.
REIMPORT_OPTION_KEY = 'reimport'


class ImporterDriver(base_driver.BaseDriver):
  """Driver for Importer."""

  def _import_artifact(self, artifact_uri: Text,
                       artifact_properties: Dict[Text, Any], reimport: bool,
                       destination_channel: types.Channel) -> types.Artifact:
    """Imports external resource in MLMD."""
    absl.logging.info('Processing source uri: %s, properties: %s' %
                      (artifact_uri, artifact_properties))

    unfiltered_previous_artifacts = self._metadata_handler.get_artifacts_by_uri(
        artifact_uri)
    # Filter by property values.
    previous_artifacts = []
    for candidate_mlmd_artifact in unfiltered_previous_artifacts:
      is_candidate = True
      candidate_artifact = destination_channel.type()
      candidate_artifact.set_mlmd_artifact(candidate_mlmd_artifact)
      for key, value in artifact_properties.items():
        if getattr(candidate_artifact, key) != value:
          is_candidate = False
          break
      if is_candidate:
        previous_artifacts.append(candidate_mlmd_artifact)

    result = destination_channel.type()
    result.uri = artifact_uri
    for key, value in artifact_properties.items():
      setattr(result, key, value)

    # If any registered artifact with the same uri also has the same
    # fingerprint and user does not ask for re-import, just reuse the latest.
    # Otherwise, register the external resource into MLMD using the type info
    # in the destination channel.
    if bool(previous_artifacts) and not reimport:
      absl.logging.info('Reusing existing artifact')
      result.set_mlmd_artifact(max(previous_artifacts, key=lambda m: m.id))
    else:
      self._metadata_handler.publish_artifacts([result])
      absl.logging.info('Registered new artifact: %s' % result)

    return result

  def pre_execution(
      self,
      input_dict: Dict[Text, types.Channel],
      output_dict: Dict[Text, types.Channel],
      exec_properties: Dict[Text, Any],
      driver_args: data_types.DriverArgs,
      pipeline_info: data_types.PipelineInfo,
      component_info: data_types.ComponentInfo,
  ) -> data_types.ExecutionDecision:
    output_artifacts = {
        IMPORT_RESULT_KEY: [
            self._import_artifact(
                artifact_uri=exec_properties[SOURCE_URI_KEY],
                artifact_properties=exec_properties[PROPERTIES_KEY],
                destination_channel=output_dict[IMPORT_RESULT_KEY],
                reimport=exec_properties[REIMPORT_OPTION_KEY])
        ]
    }

    output_dict[IMPORT_RESULT_KEY] = channel_utils.as_channel(
        output_artifacts[IMPORT_RESULT_KEY])

    return data_types.ExecutionDecision(
        input_dict={},
        output_dict=output_artifacts,
        exec_properties={},
        execution_id=self._register_execution(
            exec_properties={},
            pipeline_info=pipeline_info,
            component_info=component_info),
        use_cached_results=False)


class ImporterNode(base_node.BaseNode):
  """Definition for TFX ImporterNode.

  ImporterNode is a special TFX node which registers an external resource into
  MLMD
  so that downstream nodes can use the registered artifact as input.

  Here is an example to use ImporterNode:

  ...
  importer = ImporterNode(
      instance_name='import_schema',
      source_uri='uri/to/schema'
      artifact_type=standard_artifacts.Schema,
      reimport=False)
  schema_gen = SchemaGen(
      fixed_schema=importer.outputs['result'],
      examples=...)
  ...

  Attributes:
    _source_uri: the source uri to import.
    _reimport: whether or not to re-import the URI even if it already exists in
      MLMD.
  """

  DRIVER_CLASS = ImporterDriver

  def __init__(self,
               instance_name: Text,
               source_uri: Text,
               artifact_type: Type[types.Artifact],
               reimport: Optional[bool] = False,
               properties: Optional[Dict[Text, Any]] = None):
    """Init function for ImporterNode.

    Args:
      instance_name: the name of the ImporterNode instance.
      source_uri: the URI of the resource that needs to be registered.
      artifact_type: the type of the artifact to import.
      reimport: whether or not to re-import as a new artifact if the URI has
        been imported in before.
      properties: Dictionary of properties for the imported Artifact. These
        properties should be ones declared for the given artifact_type (see the
        PROPERTIES attribute of the definition of the type for details).
    """
    self._source_uri = source_uri
    self._reimport = reimport
    self._properties = properties or {}

    artifact = artifact_type()
    for key, value in self._properties.items():
      setattr(artifact, key, value)
    self._output_dict = {
        IMPORT_RESULT_KEY:
            types.Channel(type=artifact_type, artifacts=[artifact])
    }

    super(ImporterNode, self).__init__(instance_name=instance_name)

  def to_json_dict(self) -> Dict[Text, Any]:
    # TODO(b/145622586): Consider changing the keys to be named constants.
    return {
        '_instance_name': self._instance_name,
        '_output_dict': self._output_dict,
        '_reimport': self._reimport,
        '_source_uri': self._source_uri,
        '_properties': self._properties,
        'driver_class': self.driver_class,
        'executor_spec': self.executor_spec,
    }

  @property
  def inputs(self) -> node_common._PropertyDictWrapper:  # pylint: disable=protected-access
    return node_common._PropertyDictWrapper({})  # pylint: disable=protected-access

  @property
  def outputs(self) -> node_common._PropertyDictWrapper:  # pylint: disable=protected-access
    return node_common._PropertyDictWrapper(self._output_dict)  # pylint: disable=protected-access

  @property
  def exec_properties(self) -> Dict[Text, Any]:
    return {
        SOURCE_URI_KEY: self._source_uri,
        REIMPORT_OPTION_KEY: self._reimport,
        PROPERTIES_KEY: self._properties,
    }
