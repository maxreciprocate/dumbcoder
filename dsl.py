from operator import mul, add
from numpy import zeros, ones, empty, array
from functools import partial, reduce
from copy import deepcopy

class Delta:
    def __init__(self, head, type=None, tailtypes=None, tails=None, repr=None, hiddentail=None, arrow=None, ishole=False, isarg=False):
        self.head = head
        self.tails = tails
        self.tailtypes = tailtypes
        self.type = type
        self.ishole = ishole
        self.isarg = isarg

        if arrow:
            self.arrow = arrow
            self.type = arrow
        else:
            if tailtypes:
                self.arrow = (tuple(tailtypes), type)
            else:
                self.arrow = type

        self.hiddentail = hiddentail

        if repr is None:
            repr = str(head)

            if not self.ishole and not self.isarg and type == str:
                repr = f"'{repr}'"

        self.repr = repr
        self.idx = 0

    def __call__(self):
        if self.tails is None:
            return self.head

        if self.hiddentail:
            body = deepcopy(self.hiddentail)

            for tidx, tail in enumerate(self.tails):
                # arg in hiddentail should only match itself for replacement
                body = replace_hidden(body, Delta(f'${tidx}', isarg=True, type=tail.type), tail)

            return body()

        tails = []
        for a in self.tails:
            if isinstance(a, Delta):
                tails.append(a())
            else:
                tails.append(a)

        return self.head(*tails)

    def balance(self):
        if not self.tails:
            return self

        if not any(map(isterminal, self.tails)):
            self.tails = sorted(self.tails, key=str)

        if self.hiddentail:
            self.hiddentail.balance()

        for tail in self.tails:
            tail.balance()

        return self

    def __eq__(self, other):
        if other is None:
            return False

        if not isinstance(other, Delta):
            return False

        return isequal(self, other)

    def __hash__(self):
        return hash(repr(self))

    def __repr__(self):
        if self.tails is None or len(self.tails) == 0:
            return f'{self.repr}'
        else:
            tails = self.tails

        return f'({self.repr} {" ".join(map(str, tails))})'

def isterminal(d: Delta) -> bool:
    if d.tailtypes == None:
        return True

    if d.tails is None or len(d.tails) == 0:
        return False

    for tail in d.tails:
        if not isterminal(tail):
            return False

    return True


def length(tree: Delta) -> int:
    if not tree:
        return 0

    if not tree.tails:
        return 1

    return 1 + sum(map(length, tree.tails))

def countholes(tree: Delta) -> int:
    if not tree:
        return 0

    if tree.ishole:
        return 1

    if not tree.tails:
        return 0

    return sum(map(countholes, tree.tails))


def getdepth(tree: Delta) -> int:
    if tree.tails is None or len(tree.tails) == 0:
        return 0

    out = 0
    for tail in tree.tails:
        out = max(out, 1 + getdepth(tail))

    return out


def getast(expr):
    ast = []
    idx = 0

    while idx < len(expr):
        if expr[idx] == '(':
            nopen = 1
            sidx = idx

            while nopen != 0:
                idx += 1
                if expr[idx] == '(':
                    nopen += 1
                if expr[idx] == ')':
                    nopen -= 1

            ast.append(getast(expr[sidx+1:idx]))

        elif not expr[idx] in "() ":
            se_idx = idx
            idx += 1

            while idx < len(expr) and not expr[idx] in "() ":
                idx += 1

            ast.append(expr[se_idx:idx])

        elif expr[idx].isdigit():
            sidx = idx

            out = ''
            nopen = 1
            while idx < len(expr) and expr[idx].isdigit():
                out += expr[idx]
                idx += 1

            ast.append(out)
            # for the next ) or something else
            idx -= 1

        elif not expr[idx] in [' ', ')']:
            ast.append(expr[idx])

        idx += 1

    if isinstance(ast[0], list):
        return ast[0]

    return ast

def todelta(D, ast):
    if not isinstance(ast, list):
        if ast.startswith('$'):
            return Delta(ast)

        if (idx := D.index(ast)) is None:
            raise ValueError(f"what's a {ast}?")

        return D[idx]

    newast = []
    idx = 0
    while idx < len(ast):
        d = todelta(D, ast[idx])

        args = []

        idx += 1
        while idx < len(ast):
            args.append(todelta(D, ast[idx]))
            idx += 1

        if len(args) > 0:
            d.tails = args

        newast.append(d)

        idx += 1

    return newast[0]

def tr(D, expr):
    return todelta(D, getast(expr))

def isequal(n1, n2):
    if n1.ishole or n2.ishole:
        return n1.type == n2.type

    if n1.isarg and n2.isarg:
        return n1.type == n2.type

    if n1.head == n2.head:
        # 26 no kids
        if not n1.tails and not n2.tails:
            return True

        if not n1.tails or not n2.tails:
            return False

        if len(n1.tails) != len(n2.tails):
            return False

        for t1, t2 in zip(n1.tails, n2.tails):
            if not isequal(t1, t2):
                return False

        return True

    return False

def extract_matches(tree, treeholed):
    """
    given a healthy tree, find in it part covering holes in a given treeholed
    return pairs of holes and covered parts
    """
    if not tree or not treeholed:
        return []

    if treeholed.ishole or treeholed.isarg:
        return [(treeholed.head, deepcopy(tree))]

    out = []
    if not tree.tails:
        return []

    # assert len(tree.tails) == len(treeholed.tails)

    for tail, holedtail in zip(tree.tails, treeholed.tails):
        out += extract_matches(tail, holedtail)

    return out


def replace_hidden(tree, arg, tail):
    if isequal(tree, arg):
        return deepcopy(tail)

    if not tree.tails:
        return tree

    qq = [tree]
    while len(qq) > 0:
        n = qq.pop(0)
        if not n.tails: continue

        for idx, nt in enumerate(n.tails):
            if isequal(nt, arg):
                n.tails[idx] = tail
                break
            else:
                qq.append(nt)

    return tree

def replace(tree, matchbranch, newbranch):
    if isequal(tree, matchbranch):
        branch = deepcopy(newbranch)

        if not tree.tails:
            return branch

        args = {arg: tail for arg, tail in extract_matches(tree, matchbranch)}
        if len(args) > 0:
            branch.tails = list(args.values())

        return branch

    qq = [tree]
    while len(qq) > 0:
        n = qq.pop(0)
        if not n.tails: continue

        for i in range(len(n.tails)):
            if isequal(n.tails[i], matchbranch):
                branch = deepcopy(newbranch)
                args = {arg: tail for arg, tail in extract_matches(n.tails[i], matchbranch)}
                branch.tails = list(args.values())
                n.tails[i] = branch
            else:
                qq.append(n.tails[i])

    return tree

# d.type $ has property of wildcard matching
# making it impossible to modify hiddentails
def freeze(tree: Delta):
    if tree.ishole:
        tree.ishole = False
        tree.isarg = True

    if tree.hiddentail:
        freeze(tree.hiddentail)

    if tree.tails:
        for tail in tree.tails:
            freeze(tail)

def normalize(tree):
    if tree.hiddentail:
        ht = normalize(deepcopy(tree.hiddentail))

        if tree.tails:
            for tidx, tail in enumerate(tree.tails):
                replace_hidden(ht, Delta(f'${tidx}', isarg=True, type=tail.type), normalize(tail))

        return ht

    qq = [tree]
    while len(qq) > 0:
        n = qq.pop(0)

        if not n.tails:
            continue

        for idx in range(len(n.tails)):
            if n.tails[idx].hiddentail:
                tails = n.tails[idx].tails
                n.tails[idx] = normalize(deepcopy(n.tails[idx].hiddentail))

                if tails:
                    for tidx, tail in enumerate(tails):
                        n.tails[idx] = replace_hidden(n.tails[idx], Delta(f'${tidx}', isarg=True, type=tail.type), normalize(tail))
            else:
                qq.append(normalize(n.tails[idx]))

    return tree


# not reentrant
def typize(tree: Delta):
    "replace each hole with $arg, returning all $arg's types"
    qq = [tree]
    tailtypes = []
    z = 0

    while len(qq) > 0:
        n = qq.pop(0)

        if not n.tails:
            continue

        for idx in range(len(n.tails)):
            # is this a hole?
            if n.tails[idx].ishole:
                type = n.tails[idx].type
                tailtypes.append(type)

                # need to hole it for next tree replacement
                n.tails[idx] = Delta(f'${z}', ishole=True, type=type)
                z += 1
            else:
                qq.append(n.tails[idx])

    return tailtypes

def showoff_types(tree: Delta):
    qq = [tree]
    types = set()

    while len(qq) > 0:
        n = qq.pop(0)

        types.add(n.type)

        if not n.tails:
            continue

        for t in n.tails:
            qq.append(t)

    return types

def comp(n1, n2):
    if isequal(n1, n2):
        return deepcopy(n1)

    if not n1.tails or not n2.tails:
        return False

    for c1, c2 in zip(n1.tails, n2.tails):
        for out in [comp(n1, c2), comp(n2, c1), comp(c1, c2)]:
            if out:
                return out

            return False

def showoff_kids(tree):
    if not tree.tails:
        return

    yield str(tree)

    for tail in tree.tails:
        yield from showoff_kids(tail)

# also know as filter
def flatten(xs):
    return reduce(lambda acc, x: acc + x if x else acc, xs, [])

def alld(tree):
    "enumerate all heads in tree"
    if not tree.tails:
        return [tree]

    heads = [tree]

    for t in tree.tails:
        heads.extend(alld(t))

    return heads
