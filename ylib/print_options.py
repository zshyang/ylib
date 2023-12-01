'''
modified from
https://github.com/mengweiren/longitudinal-representation-learning/blob/main/src/options/base_options.py

author
    zhangsihao yang
logs
    2023-09-22
        file created
    2023-10-19
        add group print options
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
    for group in parser._action_groups:

        # sort actions in group
        sorted_actions = sorted(group._group_actions, key=lambda x: x.dest)

        # skip if there is no action in the group
        if len(sorted_actions) == 0:
            continue

        # print group title
        len_star = 45 - len(group.title)
        message += f"<{group.title}> {'+'*len_star}\n"

        # print options
        for action in sorted_actions:
            comment = ''
            default = parser.get_default(action.dest)
            value = getattr(args, action.dest, None)
            if value != default:
                comment = f"\t[default: {default}]"
            message += f'{str(action.dest):>25}: {str(value):<30}{comment}\n'
        message += '\n'

    message += '----------------- End -------------------'

    return message
