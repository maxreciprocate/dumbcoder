from operator import mul, add
from numpy import zeros, ones, empty, array
from functools import partial, reduce


class Delta:
    def __init__(self, head, type, tailtypes=None, tails=None):
        self.head = head
        self.tails = tails
        self.tailtypes = tailtypes
        self.type = type
        self.idx = 0

    def __call__(self):
        if self.tails is None:
            return self.head

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

        for tail in self.tails:
            tail.balance()

        return self

    def __repr__(self):
        if self.head == mul:
            head = '*'
        elif self.head == add:
            head = '+'
        elif isinstance(self.head, str):
            head = f"'{self.head}'"
        else:
            head = self.head

        if self.tails is None:
            return f'{head}'
        else:
            tails = self.tails

        return f'({head} {" ".join(map(str, tails))})'

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

    if not tree.tails or len(tree.tails) == 0:
        return 1

    out = 1
    for tail in tree.tails:
        out += length(tail)

    return out


def getdepth(tree: Delta) -> int:
    if tree.tails is None or len(tree.tails) == 0:
        return 0

    out = 0
    for tail in tree.tails:
        out = max(out, 1 + depth(tail))

    return out

def typeof(ast):
    if ast.head not in ops:
        return type(ast.head)

    t1 = typeof(ast.tails[0])
    t2 = typeof(ast.tails[1])

    if t1 == int and t2 == int:
        return int

    if t1 == str or t2 == str:
        return str

def getast(expr):
    return parse(expr)

def parse(expr):
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

            ast.append(parse(expr[sidx+1:idx]))

        elif expr[idx] == "'":
            idx += 1
            sidx = idx

            while expr[idx].isdigit():
                idx += 1

            ast.append(expr[sidx:idx])

        elif expr[idx].isdigit():
            sidx = idx

            out = ''
            nopen = 1
            while idx < len(expr) and expr[idx].isdigit():
                out += expr[idx]
                idx += 1

            ast.append(int(out))
            # for the next ) or something else
            idx -= 1

        elif not expr[idx] in [' ', ')']:
            ast.append(expr[idx])

        idx += 1

    if isinstance(ast[0], list):
        return ast[0]

    return ast


def todelta(ast):
    if not isinstance(ast, list):
        if ast == '*':
            return Delta(mul, str)
        if ast == '+':
            return Delta(add, int)

        return Delta(ast, type(ast))

    newast = []
    idx = 0
    while idx < len(ast):
        d = todelta(ast[idx])

        args = []

        idx += 1
        while idx < len(ast):
            args.append(todelta(ast[idx]))
            idx += 1

        if len(args) > 0:
            d.tails = args

        newast.append(d)

        idx += 1

    return newast[0]

def tr(expr):
    return todelta(getast(expr))

def isequal(n1, n2):
    if n1.head == n2.head:
        if not n1.tails and not n2.tails:
            return True

        if not n1.tails or not n2.tails:
            return False

        return isequal(n1.tails[0], n2.tails[0]) and isequal(n1.tails[1], n2.tails[1])
    return False


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

def flatten(xs):
    return reduce(lambda acc, x: acc + x, xs, [])

def alld(tree):
    "enumerate all heads in tree"
    if not tree.tails:
        return [tree]

    heads = [tree]

    for t in tree.tails:
        heads.extend(alld(t))

    return heads
