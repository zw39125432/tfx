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
"""TFX InfraValidator executor definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import os

from absl import logging
from typing import cast, Any, Dict, List, Text

from google.protobuf import json_format
from tfx import types
from tfx.components.base import base_executor
from tfx.components.infra_validator import request_builder
from tfx.components.infra_validator import types as infra_validator_types
from tfx.components.infra_validator.model_server_runners import factory
from tfx.proto import infra_validator_pb2
from tfx.types import artifact_utils
from tfx.types import standard_artifacts
from tfx.utils import io_utils
from tfx.utils import path_utils

Model = standard_artifacts.Model
Examples = standard_artifacts.Examples
TensorFlowServingRpcKind = infra_validator_pb2.TensorFlowServingRpcKind

TENSORFLOW_SERVING = 'tensorflow_serving'
# Filename of infra blessing artifact on succeed.
BLESSED = 'INFRA_BLESSED'
# Filename of infra blessing artifact on fail.
NOT_BLESSED = 'INFRA_NOT_BLESSED'
_DEFAULT_MAX_EXAMPLES = 100
_DEFAULT_EXAMPLES_SPLIT = 'eval'


class Executor(base_executor.BaseExecutor):
  """TFX infra validator executor."""

  def Do(self, input_dict: Dict[Text, List[types.Artifact]],
         output_dict: Dict[Text, List[types.Artifact]],
         exec_properties: Dict[Text, Any]) -> None:
    """Contract for running InfraValidator Executor.

    Args:
      input_dict:
        - `model`: Single `Model` artifact that we're validating.
        - `examples`: `Examples` artifacts to be used for test requests.
      output_dict:
        - `blessing`: Single `InfraBlessing` artifact containing the validated
          result. It is an empty file with the name either of INFRA_BLESSED or
          INFRA_NOT_BLESSED.
      exec_properties:
        - `serving_spec`: Serialized `ServingSpec` configuration.
        - `validation_spec`: Serialized `ValidationSpec` configuration.
        - `request_spec`: Serialized `RequestSpec` configuration.
    """
    model = artifact_utils.get_single_instance(input_dict['model'])
    examples = (artifact_utils.get_single_instance(input_dict['examples'])
                if 'examples' in input_dict
                else None)
    blessing = artifact_utils.get_single_instance(output_dict['blessing'])

    serving_spec = infra_validator_pb2.ServingSpec()
    json_format.Parse(exec_properties['serving_spec'], serving_spec)
    validation_spec = infra_validator_pb2.ValidationSpec()
    json_format.Parse(exec_properties['validation_spec'], validation_spec)
    if 'request_spec' in exec_properties:
      request_spec = infra_validator_pb2.RequestSpec()
      json_format.Parse(exec_properties['request_spec'], request_spec)
    else:
      request_spec = None

    model_name = os.path.basename(
        os.path.dirname(path_utils.serving_model_path(model.uri)))
    if examples and request_spec:
      logging.info('InfraValidator will be run in LOAD_AND_QUERY mode.')
      requests = self._BuildRequests(model_name, cast(Examples, examples),
                                     request_spec)
    else:
      logging.info('InfraValidator will be run in LOAD_ONLY mode.')
      requests = None

    runners = factory.create_model_server_runners(
        cast(Model, model), serving_spec)

    # TODO(jjong): Make logic parallel.
    for runner in runners:
      with _defer_stop(runner):
        logging.info('Starting %s.', repr(runner))
        runner.Start()

        # Check model is successfully loaded.
        if not runner.WaitUntilModelAvailable(
            timeout_secs=validation_spec.max_loading_time_seconds):
          logging.info('Failed to load model in %s; marking as not blessed.',
                       repr(runner))
          self._Unbless(blessing)
          return

        # Check model can be successfully queried.
        if requests:
          if not runner.client.IssueRequests(requests):
            logging.info('Failed to query model in %s; marking as not blessed.',
                         repr(runner))
            self._Unbless(blessing)
            return

    logging.info('Model passed infra validation; marking model as blessed.')
    self._Bless(blessing)

  @staticmethod
  def _BuildRequests(
      model_name: Text,
      examples: Examples,
      request_spec: infra_validator_pb2.RequestSpec
  ) -> List[infra_validator_types.Request]:
    """Build a list of request protos to be queried aginst the model server.

    Args:
      model_name: Name of the model. For example, tensorflow `SavedModel` is
        saved under directory `{model_name}/{version}`. The same directory
        structure is reused in a tensorflow serving, and you need to specify
        `model_name` in the request to access it.
      examples: An `Examples` artifact which contains gzipped TFRecord file
        containing `tf.train.Example`.
      request_spec: A `RequestSpec` config.

    Returns:
      A list of request protos.
    """
    split_name = request_spec.split_name or _DEFAULT_EXAMPLES_SPLIT
    builder = request_builder.RequestBuilder(
        max_examples=request_spec.max_examples or _DEFAULT_MAX_EXAMPLES,
        model_name=model_name
    )
    logging.info('InfraValidator is using "%s" split.', split_name)
    builder.ReadFromExamplesArtifact(examples, split_name=split_name)

    kind = request_spec.WhichOneof('serving_binary')
    if kind == TENSORFLOW_SERVING:
      spec = request_spec.tensorflow_serving
      if spec.signature_name:
        builder.SetSignatureName(spec.signature_name)
      if spec.rpc_kind == TensorFlowServingRpcKind.CLASSIFY:
        return builder.BuildClassificationRequests()
      elif spec.rpc_kind == TensorFlowServingRpcKind.REGRESS:
        return builder.BuildRegressionRequests()
      else:
        raise ValueError('Invalid TensorFlowServingRpcKind: {}'.format(
            spec.rpc_kind))
    else:
      raise ValueError('Invalid RequestSpec {}'.format(request_spec))

  def _Bless(self, blessing):
    io_utils.write_string_file(os.path.join(blessing.uri, BLESSED), '')
    blessing.set_int_custom_property('blessed', 1)

  def _Unbless(self, blessing):
    io_utils.write_string_file(os.path.join(blessing.uri, NOT_BLESSED), '')
    blessing.set_int_custom_property('blessed', 0)


@contextlib.contextmanager
def _defer_stop(stoppable):
  try:
    yield
  finally:
    logging.info('Stopping %s.', repr(stoppable))
    stoppable.Stop()
