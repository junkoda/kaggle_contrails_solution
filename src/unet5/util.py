from typing import Tuple


def get_range(ra: str, n: int) -> Tuple[int, int]:
    """
    istart, iend = get_range('1:10', n)

    Args:
      ra (str or None): 1, 1:, 5:10
      n (int): default iend when it is omitted, ra = '1:'
    """

    if ra is None:
        return 0, n
    elif ra.isnumeric():
        # Single number
        i = int(ra)
        return i, i + 1
    elif ':' in ra:
        v = ra.split(':')
        assert len(v) == 2
        istart = 0 if v[0] == '' else int(v[0])
        iend = n if v[1] == '' else int(v[1])
        return istart, iend
    else:
        raise ValueError('Failed to parse range: {}'.format(ra))


def as_list(s: str, *, dtype=int):
    """
    Parse string of numbers to list of numbers

    Arg:
      s (str): '1,2,4...7' => [1, 2, 4, 5, 6, 7]
    """
    ret = []
    if not isinstance(s, str):
        return [dtype(s), ]

    for seg in s.split(','):
        if dtype is int and '...' in seg:
            start, last = map(int, seg.split('...'))
            for i in range(start, last):
                ret.append(dtype(i))
        else:
            ret.append(dtype(seg))

    return ret
