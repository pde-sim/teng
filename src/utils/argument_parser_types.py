import argparse
import ast
import json


def boolargparse(value):
    if value == True or value == 1 or value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value == False or value == 0 or value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f'Invalid value: {value}. Boolean value expected')


def jsonargparse(string):
    try:
        return json.loads(string)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid JSON string: {string}") from e


### use with caution
# def evalargparse(string):
#     if string.isidentifier() and not iskeyword(string):
#         return eval(string)
#     else:
#         raise argparse.ArgumentTypeError(f'Invalid value: {string}. Must be an identifier and must not be a keyword.')

# ArgsKwargsParse = namedtuple('ParsedArguments', ['args', 'kwargs'])

# class ArgsParseAction(argparse.Action):
#     def __call__(self, parser, namespace, values, option_string=None):
#         args = []
#
#         if values:
#             # Flatten and split the arguments by both comma and space
#             items = [item.strip() for sublist in values for item in sublist.replace(',', ' ').split() if item.strip()]
#
#             for item in items:
#                 args.append(ast.literal_eval(item.strip()))
#
#         setattr(namespace, self.dest, args)

class ArgsKwargsParseAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        args = []
        kwargs = {}

        if values:
            # Flatten and split the arguments by both comma and space
            items = [item.strip() for sublist in values for item in sublist.replace(',', ' ').split() if item.strip()]

            for item in items:
                if "=" in item:
                    key, value = item.split("=", 1)
                    kwargs[key.strip()] = ast.literal_eval(value.strip())
                else:
                    args.append(ast.literal_eval(item.strip()))

        setattr(namespace, self.dest, (args, kwargs))
