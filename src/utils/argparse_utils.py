def create_dict_from_argparse_remainder(remainder_args):
    kwargs = {}
    if remainder_args is not None:
        if len(remainder_args) % 2 != 0:
            raise ValueError('Extra args must be key-value pairs')
    for i in range(0, len(remainder_args), 2):
        kwargs[remainder_args[i]] = remainder_args[i + 1]

    return kwargs
