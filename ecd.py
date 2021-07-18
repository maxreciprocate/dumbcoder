from operator import mul, add

import random
import numpy as np
from numpy.random import rand, randint, randn, normal
from numpy import zeros, ones, empty, array
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass

from typing import Optional, Union, List, NamedTuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, as_tensor, from_numpy

from itertools import chain
from functools import partial, reduce, lru_cache

from tqdm import tqdm, trange
from copy import deepcopy
from time import time, sleep

from dsl import *
from gpt import *

class Deltas:
    def __init__(self, core):
        self.core = core
        self.invented = []

        self.infer()

    def add(self, d: Delta, terminal=True):
        if not terminal:
            self.invented.append(d)
            self.infer()
            return

        out = d()

        self.invented.append(Delta(out, type(out)))
        self.infer()

    def infer(self):
        self.ds = self.core + self.invented
        self.terminals = array([isterminal(d) for d in self.ds])

        self.types = [d.type for d in self.ds]
        self.childtypes = [d.tailtypes for d in self.ds]

        self.bytype_terminal = defaultdict(list)
        self.bytype = defaultdict(list)

        for i, d in enumerate(self.ds):
            if not d.tailtypes:
                self.bytype_terminal[d.type].append(i)

            self.bytype[d.type].append(i)

        for idx, d in enumerate(self.ds):
            d.idx = idx

    def logp(self, Q, d):
        if d.tails is None:
            return Q[self.index(d)]

        out = 0
        for tail in d.tails:
            out += self.logp(Q, tail)

        return Q[self.index(d)] + out

    def __iter__(self):
        return chain(self.core + self.invented)

    def __getitem__(self, idx):
        # always careful, always on the watch
        return deepcopy(self.ds[idx])

    def __len__(self):
        return len(self.ds)

    def __repr__(self):
        return f"{self.core} + {self.invented}"

    def __contains__(self, d):
        if d in self.ds:
            return True

        if isinstance(d, Delta):
            outd = d()
        else:
            outd = d

        od = [d() for d in self.ds if not d.tailtypes]

        return outd in od


    def index(self, d):
        for idx, _d in enumerate(self.ds):
            if d.head == _d.head:
                return idx

        return None

    def reset(self):
        self.invented = []
        self.infer()

@dataclass
class Gen:
    # logp of the saved enumeration
    logp: float
    # generator of the current branch expansion or a frozen branch
    generator: Union['generator', Delta]
    # logp of the current expanded branch (self.branch_logp > self.logp)
    branch_logp: float
    # generators for the rest of the branches
    next_generators: List['generator']

    def __repr__(self):
        return f"({self.logp:.2f}/{self.branch_logp:.2f}) [{' '.join(ap(str, self.next_generators))}]"


def makepaths(D, logits):
    Paths = [[] for i in range(len(D))]
    Paths_terminal = [[] for i in range(len(D))]

    for d in D:
        if not d.tailtypes:
            continue

        for tidx, tailtype in enumerate(d.tailtypes):
            if isinstance(logits, th.Tensor):
                ps = logits.clone()
            else:
                ps = deepcopy(logits)

            # limit by type
            possibles = D.bytype[tailtype]

            for idx in range(len(ps)):
                if idx not in possibles:
                    ps[idx] = -np.inf

            Paths[d.idx].append(F.log_softmax(ps, -1))

            # permit leafs
            possibles_terminal = D.bytype_terminal[tailtype]

            if isinstance(logits, th.Tensor):
                ps = logits.clone()
            else:
                ps = deepcopy(logits)

            for idx in range(len(ps)):
                if idx not in possibles_terminal:
                    ps[idx] = -np.inf

            Paths_terminal[d.idx].append(F.log_softmax(ps, -1))

    return Paths, Paths_terminal


def creategens(D, sources, paths, paths_terminal, maxdepth):
    if len(sources) == 0:
        return []

    source, *nextsources = sources

    out = []
    for idx, logp in enumerate(source):
        branchgen = denumerate(D, D[idx], logp, paths, paths_terminal, maxdepth)
        gen = Gen(logp, branchgen, logp, creategens(D, nextsources, paths, paths_terminal, maxdepth))

        out.append(gen)

    return out

def denumerate(D, n, nlogp, paths, paths_terminal, maxdepth=10, verb=False):
    if not n.tailtypes:
        yield nlogp, n
        return

    sources = paths[n.idx] if maxdepth > 1 else paths_terminal[n.idx]

    gensources = creategens(D, sources, paths, paths_terminal, maxdepth - 1)

    exhausted = False
    while not exhausted:
        n.tails = []
        gens = []
        idx = 0

        generators = gensources

        logps = []
        while len(generators) > 0:
            tail = None
            while tail is None:
                gen = None
                maxlogp = -np.inf
                for g in generators:
                    if g.logp > maxlogp:
                        maxlogp = g.logp
                        gen = g

                if maxlogp == -np.inf:
                    exhausted = True
                    g.logp = -np.inf

                    if idx > 0:
                        # exhausted left tail
                        gens[-1].logp = -np.inf

                    break

                if isinstance(gen.generator, Delta):
                    logp, tail = gen.branch_logp, gen.generator
                    logps.append(logp)
                else:
                    try:
                        logp, tail = next(gen.generator)

                        # not the rightest
                        if len(gen.next_generators) > 0:
                            frozen = Gen(logp, deepcopy(tail), logp, creategens(D, sources[idx+1:], paths, paths_terminal, maxdepth-1))
                            generators.append(frozen)

                            # don't want to try this one next
                            gen.logp = logp - 1e-6
                            # retry
                            tail = None
                        else:
                            logps.append(logp)

                    except StopIteration:
                        gen.logp = -np.inf

            if exhausted:
                break

            gens.append(gen)
            idx += 1

            n.tails.append(tail)

            generators = gen.next_generators
            if len(generators) == 0:
                break

        if exhausted:
            if idx > 0:
                exhausted = False

            continue

        for gen, logp in zip(reversed(gens), np.cumsum(logps[::-1])):
            gen.logp = logp

        yield nlogp + gens[0].logp, n


def expand(root: Delta, node: Delta, depth=0):
    deltas = D.bytype_terminal if depth <= 1 else D.bytype

    if node.tailtypes is None:
        yield deepcopy(root)
        return

    for lc in deltas[node.tailtypes[0]]:
        lchild = deepcopy(D[lc])

        if isterminal(lchild):
            trees = [None]
        else:
            trees = expand(root, lchild, depth - 1)

        for _ in trees:
            for rc in deltas[node.tailtypes[1]]:
                rchild = deepcopy(D[rc])

                node.tails = [lchild, rchild]

                yield deepcopy(root)

                if not isterminal(rchild):
                    yield from expand(root, rchild, depth - 1)


def solve(X, D, depth=3):
    solutions = {x: None for x in X}

    sources = []
    for d in D:
        if d.type == str:
            root = deepcopy(d)
            sources.append(expand(root, root, depth=depth))

    cnt = 0
    stime = time()

    for tree in chain.from_iterable(sources):
        if not isterminal(tree):
            continue

        cnt += 1
        out = tree()

        if out in X:
            if solutions[out] is None or length(tree) < length(solutions[out]):
                solutions[out] = tree

    took = time() - stime
    print(f'total: {cnt}, took: {took:.0f}s, iter: {cnt/took:.0f}/s')
    print(f'solved: {sum(s is not None for s in solutions.values())}/{len(solutions)}')
    return solutions

def needle(D, n, paths, paths_terminal, depth=0):
    if n.tailtypes is None:
        return

    source = paths_terminal if depth <= 1 else paths
    n.tails = []

    for path in source[n.idx]:
        nn = deepcopy(D[sample(path)])

        n.tails.append(nn)
        needle(D, nn, paths, paths_terminal, depth - 1)


def newtree(D, type, paths, paths_terminal, depth=6, q=None):
    if q is None:
        q = th.ones(len(D))

    if q.requires_grad:
        q = q.detach()

    q = q.flatten()
    qroot = deepcopy(q)

    for i in range(len(q)):
        if i not in D.bytype[type]:
            qroot[i] = -np.inf

    qroot = F.softmax(qroot, -1)

    root = D[sample(qroot)]
    tree = deepcopy(root)

    needle(D, tree, paths, paths_terminal, depth=depth)

    return tree

def solve_needle(X, D, Q, solutions=None, depth=6, ntries=100_000):
    print(f'{len(D)=}')

    if solutions is None:
        solutions = {x: None for x in X}

    cnt = 0
    stime = time()
    notsolved = sum([s is None for s in solutions.values()])

    paths, paths_terminal = makepaths(D, Q)

    while True:
        tree = newtree(D, type(X[0]), paths, paths_terminal, q=Q)
        try:
            out = tree()
        except TypeError:
            print(f"what is this {tree=}?")

        cnt += 1

        if out in X:
            if solutions[out] is None:
                notsolved -= 1
                print(f'[{cnt:6d}] caught {out}')

            if solutions[out] is None or length(tree) < length(solutions[out]):
                solutions[out] = tree

        if cnt > ntries:
            break


    took = time() - stime
    print(f'total: {cnt}, took: {took:.0f}s, iter: {cnt/took:.0f}/s')
    print(f'solved: {sum(s is not None for s in solutions.values())}/{len(solutions)}')
    return solutions, notsolved


def solve_enumeration(X, D, Q, solutions=None, maxdepth=6, ntries=100_000):
    print(f'{len(D)=}')

    if solutions is None:
        solutions = {x: None for x in X}

    cnt = 0
    stime = time()
    notsolved = sum([s is None for s in solutions.values()])

    requested_type = type(X[0][0])
    # add an ephermal production
    D = Deltas([Delta(None, None, [requested_type])] + D.ds)
    Q = th.hstack((tensor([0]), Q))

    ephermal = deepcopy(D[0])
    paths, paths_terminal = makepaths(D, Q)

    for logp, tree in denumerate(D, ephermal, 0, paths, paths_terminal, maxdepth=maxdepth):
        # unwrap ephermal root
        tree = tree.tails[0]

        try:
            out = tree()
        except TypeError:
            print(f"what is this {tree=}?")

        cnt += 1

        if out in X:
            if solutions[out] is None:
                notsolved -= 1
                print(f'[{cnt:6d}] caught {out}')

            if solutions[out] is None or length(tree) < length(solutions[out]):
                solutions[out] = deepcopy(tree)

        if cnt > ntries:
            break


    took = time() - stime
    print(f'total: {cnt}, took: {took:.0f}s, iter: {cnt/took:.0f}/s')
    print(f'solved: {sum(s is not None for s in solutions.values())}/{len(solutions)}')
    return solutions, notsolved

def replace(tree, oldbranch, newbranch):
    "replace given subtree with a new one"
    if isequal(tree, oldbranch):
        tree = newbranch

    if not tree.tails:
        return

    for idx in range(len(tree.tails)):
        if isequal(tree.tails[idx], oldbranch):
            tree.tails[idx] = newbranch

        replace(tree.tails[idx], oldbranch, newbranch)


def kcompress(D, trees):
    while True:
        ds = flatten(list(map(lambda tree: list(showoff_kids(tree)), trees)))
        count = Counter(ds)
        most_common = count.most_common()

        totall = sum(map(length, trees))
        mink = totall
        nd = None

        for d, c in most_common:
            if c < 2:
                continue

            d = tr(d)

            topkalon = totall - c * length(d) + c + length(d)
            if topkalon < mink:
                mink = topkalon
                nd = d

        if nd is None:
            return trees

        if d in D:
            return trees

        print(f'selecting {nd() if nd else ""} {nd} with {mink/totall:.2f} of {count[str(nd)]}')

        D.add(nd)

        for tree in trees:
            replace(tree, nd, Delta(nd(), type(nd())))


def tc(x):
    return tensor([int(c) for c in x])

class RecognitionModel(nn.Module):
    def __init__(self, nd):
        super().__init__()
        self.gpt = YOGPT(vocabsize=2, nheads=8, nembd=64, ntoks=100, nlayers=8)
        self.head = nn.Linear(self.gpt.nembd, nd)

    def forward(self, x):
        return self.head(self.gpt(x).mean(1))


def dream(D, soltrees=[]):
    ntoks = 100
    qmodel = RecognitionModel(len(D))
    opt = th.optim.Adam(qmodel.parameters())
    paths, paths_terminal = makepaths(D, th.ones(len(D)))

    tbar = trange(100)
    for _ in tbar:
        trees = [newtree(D, str, paths, paths_terminal, depth=7) for _ in range(10)] + soltrees
        print(trees)

        Xy = [[tree()[:ntoks], alld(tree)] for tree in trees]

        # x, i: d
        Xy = [[(xy[0], D.index(d)) for d in xy[1]] for xy in Xy]
        Xy = reduce(lambda acc, xy: acc + xy, Xy, [])

        X = [tc(xy[0]) for xy in Xy]

        q = th.vstack([qmodel(x[None]) for x in X])
        y = tensor([xy[1] for xy in Xy])

        opt.zero_grad()
        loss = F.cross_entropy(q, y)
        loss.backward()
        opt.step()

        tbar.set_description(f'{loss=:.2f}')

    return qmodel


def ECD(X, D, ntries=1e6):
    D.reset()

    sols = {x: None for x in X}
    Q = th.arange(1, len(D)+1).float()
    Q = F.log_softmax(Q, -1)

    while True:
        # explore
        sols, nunsolved = solve_enumeration(X, D, Q, sols, maxdepth=10, ntries=ntries)

        pprint(sols)
        trees = [s.balance() for s in sols.values() if s]

        # compress
        trees = kcompress(D, trees)
        print(D)

        if nunsolved == 0:
            return trees

        evald = {tree(): tree for tree in trees}
        sols = {x: None if x not in evald else evald[x] for x in X}

        # dream
        Qmodel = dream(D, trees)
        Q = Qmodel(tc(X[0])[None]).flatten().detach()
        Q = F.log_softmax(Q, -1)


if __name__ == '__main__':
    D = Deltas([
        Delta(add, int, [int, int]),
        Delta(mul, str, [str, int]),
        Delta(add, str, [str, str]),
        Delta(3, int),
        Delta(2, int),
        Delta('0', str),
        Delta('1', str),
    ])

    X = [
        "1000",
        "10001000",
        "000000000000",
        "100000000000",
        "100010000000",
        "100000001000",
        "100010001000",
    ]

    sols = ECD(X, D, ntries=10000)
    sols
