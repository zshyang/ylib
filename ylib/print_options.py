'''
modified from
https://github.com/mengweiren/longitudinal-representation-learning/blob/main/src/options/base_options.py

author
    zhangsihao yang
logs
    2023-09-22
        file created
'''
import argparse


def options_str(args: argparse.Namespace, parser) -> str:
    '''
    print and save options

    It will print both current options and default values(if different).

    inputs:
    -------
    args: argparse.Namespace
        arguments
    parser: argparse.ArgumentParser
        parser

    return:
    -------
    message: str
        message to be printed
    '''
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(args).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = f"\t[default: {default}]"
        message += f'{str(k):>25}: {str(v):<30}{comment}\n'
    message += '----------------- End -------------------'
    return message
