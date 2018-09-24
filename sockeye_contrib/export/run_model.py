# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not
# use this file except in compliance with the License. A copy of the License
# is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is distributed on
# an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Load a trained model and run it to score validation data.
"""
import argparse
import os
import shutil
import sys
import tempfile
from contextlib import ExitStack
from typing import Any, cast, Optional, Dict, List, Tuple

import mxnet as mx


def main():
    # Arguments
    params = argparse.ArgumentParser(description="Load a Sockeye MXNet model and score validation data.")
    params.add_argument("--symbol", "-s", required=True,
                        help="MXNet symbol file (usually named symbol.json).")
    params.add_argument("--params", "-p", required=True,
                        help="MXNet params file (usually named params.best).")
    params.add_argument("--validation-source", "-vs", required=True,
                        help="Validation source file.")
    params.add_argument("--validation-target", "-vt", required=True,
                        help="Validation target file.")
    params.add_argument("--batch-size", "-b", type=int, default=64,
                        help="Validation target file. Default: %(default)s.")
    args = params.parse_args()

    # Use CPU for initial testing
    ctx = mx.cpu()

    # Load symbol and params
    symbol = mx.sym.load(args.symbol)
    save_dict = mx.nd.load(args.params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(":", 1)
        if tp == "arg":
            arg_params[name] = v
        if tp == "aux":
            aux_params[name] = v

    # Create module

    # Sockeye constants
    SOURCE_NAME = "source"
    TARGET_NAME = "target"
    TARGET_LABEL_NAME = "target_label"
    LAYOUT_BATCH_MAJOR = "NCHW"

    data_names = [SOURCE_NAME, TARGET_NAME]
    label_names = [TARGET_LABEL_NAME]

    # TODO
    default_bucket_key = [100, 100]
    num_factors = 1

    provide_data = [
        mx.io.DataDesc(name=SOURCE_NAME,
                       shape=(args.batch_size, default_bucket_key[0], num_factors),
                       layout=LAYOUT_BATCH_MAJOR),
        mx.io.DataDesc(name=TARGET_NAME,
                       shape=(args.batch_size, default_bucket_key[1]),
                       layout=LAYOUT_BATCH_MAJOR)]
    provide_label = [
        mx.io.DataDesc(name=TARGET_LABEL_NAME,
                       shape=(args.batch_size, default_bucket_key[1]),
                       layout=LAYOUT_BATCH_MAJOR)]

    module = mx.mod.Module(symbol=symbol, data_names=data_names, label_names=label_names, context=ctx)
    module.bind(data_shapes=provide_data, label_shapes=provide_label, for_training=True)
    module.set_params(arg_params, aux_params)


if __name__ == "__main__":
    main()
