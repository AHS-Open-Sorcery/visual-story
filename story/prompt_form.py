# Forms prompts for GPT2 from detected elements

from collections import Counter
from typing import Iterable


def prompt(elements: Counter, tags) -> str:
    return story_of(elements, tags)


# strategy 1: "the story of "
def story_of(elements: Counter, tags) -> str:
    def pluralize(pair):
        return 'the ' + pair[0] if pair[1] == 1 else f'{pair[1]} {pair[0]}s'

    elem_exprs = list(map(pluralize, elements.items()))
    total_elem_expr = ', '.join(elem_exprs[:-1]) + ', and ' + elem_exprs[-1]
    addition = ' with ' + ' and '.join(tags) if tags else ''
    return 'The Story of ' + total_elem_expr + addition
