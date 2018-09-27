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

import argparse
import json
import logging

import mxnet as mx

import sockeye.constants as C

def main():
    params = argparse.ArgumentParser(description="Load pre-trained MXNet model and score validation data.")
    params.add_argument("--symbol", "-s", required=True,
                        help="MXNet symbol file (named `MODEL-symbol.json' or similar).")
    params.add_argument("--params", "-p", required=True,
                        help="MXNet params file (named `MODEL-CHECKPOINT.params' or similar).")
    params.add_argument("--metric", "-m", required=True,
                        help="MXNet EvalMetric config (JSON file).")
    params.add_argument("--data", "-d", required=True,
                        help="Validation data file (MXNet str->NDArray dict format).")
    params.add_argument("--label", "-l", required=True,
                        help="Validation label file (MXNet str->NDArray dict format).")
    params.add_argument("--batch-size", "-b", type=int, default=64,
                        help="Validation target file. Default: %(default)s.")
    args = params.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logging.info("Loading symbol file `%s'." % args.symbol)
    symbol = mx.sym.load(args.symbol)

    logging.info("Loading params file `%s'." % args.params)
    save_dict = mx.nd.load(args.params)
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(":", 1)
        if tp == "arg":
            arg_params[name] = v
        if tp == "aux":
            aux_params[name] = v

    logging.info("Loading metric file `%s'." % args.metric)
    with open(args.metric, "r") as inp:
        config = json.load(inp)
    metric = mx.metric.create(**config)

    logging.info("Loading data file `%s'." % args.data)
    data = mx.nd.load(args.data)

    logging.info("Loading label file `%s'." % args.label)
    label = mx.nd.load(args.label)

    logging.info("Creating data iterator with batch size %d." % args.batch_size)
    nd_iter = mx.io.NDArrayIter(data=data, label=label, batch_size=args.batch_size)

    logging.info("Creating module.")
    module = mx.mod.Module(symbol=symbol, data_names=list(data.keys()), label_names=list(label.keys()))
    module.bind(data_shapes=nd_iter.provide_data, label_shapes=nd_iter.provide_label, for_training=True)
    module.set_params(arg_params, aux_params)

    logging.info("Scoring validation data")
    for batch in nd_iter:
        module.forward(batch, is_train=False)
        module.update_metric(metric, batch.label)

    print(metric.get_name_value())


if __name__ == "__main__":
    main()
