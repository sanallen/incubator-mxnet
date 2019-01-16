from __future__ import print_function
import argparse
import mxnet as mx
import lprnet_concat_deploy as deploy_net
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Convert a trained plate model to deploy model')
    # parser.add_argument('--network', dest='network', type=str, default='legacy_pelee',
    #                     help='which network to use')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=500, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default="/opt/models/mxnet/plate/plate_lprnet/lprnet_concat_99.01/lprnet_concat", type=str)
    # parser.add_argument('--data-shape', dest='data_shape', type=int, default=320,
    #                     help='data shape')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    sym, arg_params, aux_params = mx.model.load_checkpoint(args.prefix, args.epoch)
    # pred_fc = sym.get_internals()['softmaxactivation0_output']
    net = deploy_net.get_symbol()
    # pred_fc = sym.get_internals()['softmaxactivation0']
    for symbol in net.get_internals():
        print(symbol)
    # new name
    tmp = args.prefix.rsplit('/', 1)
    save_prefix = '/deploy_'.join(tmp)
    mx.model.save_checkpoint(save_prefix, args.epoch, net, arg_params, aux_params)
    print("Saved model: {}-{:04d}.params".format(save_prefix, args.epoch))
    print("Saved symbol: {}-symbol.json".format(save_prefix))
