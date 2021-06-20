"""
Basic noising functions
"""
import random


def rand_case(s):
    return ''.join([c.lower() if random.randint(0, 1) else c.upper() for c in s])


def repeat(s, p):
    ret = []
    for c in s:
        ret.append(c)
        if c.isalpha() and random.random() < p:
            ret.extend([c] * random.randint(1, 3))
    return ''.join(ret)

