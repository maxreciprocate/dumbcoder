from operator import mul, add

import random
import numpy as np
from numpy.random import rand, randint, randn, normal
from numpy import zeros, ones, empty, array
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass

from itertools import product, repeat

from typing import Optional, Union, List, NamedTuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, as_tensor, from_numpy

from itertools import chain
from functools import partial, reduce, lru_cache

import pickle
from tqdm import tqdm, trange
from copy import deepcopy
from time import time, sleep

from dsl import *
from gpt import *

import os

ncores = os.cpu_count() // 2
import multiprocessing as mp

# ■ ~

class Deltas:
    def __init__(self, core):
        self.core = core
        self.invented = []

        self.infer()

    def add(self, d: Delta, terminal=True):
        self.invented.append(d)
        self.infer()

    def pop(self, d: Delta):
        self.invented.pop(self.index(d) - len(self.core))
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
        if isinstance(idx, str):
            if (idx := self.index(idx)) is None:
                return None

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


    def index(self, d: Union[Delta, str]):
        for idx, dd in enumerate(self.ds):
            if isinstance(d, Delta):
                if d.head == dd.head and d.type == dd.type:
                    return idx
            else:
                if d == dd.repr:
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


def makepaths(D, Q):
    Paths = [[] for i in range(len(D))]
    Paths_terminal = [[] for i in range(len(D))]

    for d in D:
        if not d.tailtypes:
            continue

        for tidx, tailtype in enumerate(d.tailtypes):
            ps = Q.clone()

            # limit by type
            possibles = D.bytype[tailtype]
            for idx in range(len(ps)):
                if idx not in possibles:
                    ps[idx] = -np.inf

            ps = F.log_softmax(ps, -1).tolist()
            Paths[d.idx].append(deepcopy(ps))

            # permit leafs
            possibles_terminal = D.bytype_terminal[tailtype]

            for idx in range(len(ps)):
                if idx not in possibles_terminal:
                    ps[idx] = -np.inf

            Paths_terminal[d.idx].append(ps)

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

        yield nlogp + gens[0].logp, deepcopy(n)


def p2enumerate(n, nlogp, prebudget, budget, maxdepth=3):
    if budget < 0 or isterminal(n):
        yield nlogp, n
        return

    sources = paths[int(maxdepth <= 1)]
    lsources, rsources = sources[n.idx]

    for lidx, llogp in enumerate(lsources):
        if budget + llogp < 0:
            continue

        for llogp, ltree in p2enumerate(D[lidx], llogp, prebudget + llogp, budget + llogp, maxdepth-1):
            for ridx, rlogp in enumerate(rsources):
                if budget + llogp + rlogp < 0:
                    continue

                for rlogp, rtree in p2enumerate(D[ridx], rlogp, prebudget + llogp + rlogp, budget + llogp + rlogp, maxdepth-1):

                    if isterminal(D[ridx]) and prebudget > 0:
                        continue

                    n.tails = [ltree, rtree]

                    yield llogp + rlogp, deepcopy(n)


def cenumerate(D, Q, tp, budget, maxdepth, cb):
    if budget[1] <= 0 or maxdepth < 0:
        return True

    for i in D.bytype[tp]:
        if -Q[i] > budget[1]:
            continue

        d = D[i]
        logp = Q[i]
        nbudget = (budget[0] + logp, budget[1] + logp)

        cenumerate_fold(D, Q, d, d.tailtypes, nbudget, logp, maxdepth - 1, cb)

def cenumerate_fold(D, Q, d, tailtypes, budget, offset, maxdepth, cb):
    if tailtypes is not None and len(tailtypes) > 0:
        tailtp = tailtypes.pop(0)

        def ccb(tail, tlogp):
            nd = deepcopy(d)
            if nd.tails is None:
                nd.tails = []

            nd.tails.append(tail)
            nbudget = (budget[0] + tlogp, budget[1] + tlogp)
            noffset = offset + tlogp

            cenumerate_fold(D, Q, nd, deepcopy(tailtypes), nbudget, noffset, maxdepth, cb)

        return cenumerate(D, Q, tailtp, (0, budget[1]), maxdepth, ccb)

    if budget[0] < 0 and 0 <= budget[1]:
        return cb(d, offset)

    return True

def groom(D, sources, alogp, budget, paths, maxdepth):
    if len(sources) == 0:
        yield alogp, []
        return

    source, *nextsources = sources

    for idx, logp in enumerate(source):
        if budget + logp < 0:
            continue

        for nlogp, tree in penumerate(D, D[idx], logp, budget + logp, paths, maxdepth-1):
            for nnlogp, nntrees in groom(D, nextsources, alogp + nlogp, budget + nlogp, paths, maxdepth-1):
                yield nnlogp, [tree] + nntrees


def penumerate(D, n, nlogp, budget, paths, maxdepth=3):
    if budget < 0 or isterminal(n):
        yield nlogp, n
        return

    sources = paths[int(maxdepth <= 1)][n.idx]

    for logp, args in groom(D, sources, nlogp, budget + nlogp, paths, maxdepth-1):
        n.tails = args
        yield logp, deepcopy(n)

def sgroom(D, sources, alogp, budget, paths, maxdepth):
    if len(sources) == 0:
        yield alogp, []
        return

    source, *nextsources = sources

    for idx, (logp, nz) in enumerate(source):
        if logp == -np.inf:
            continue

        for nlogp, tree in spenumerate(D, D[idx], nz, logp, budget + logp, paths, maxdepth-1):
            for nnlogp, nntrees in sgroom(D, nextsources, alogp + nlogp, budget + nlogp, paths, maxdepth-1):
                yield nnlogp, [tree] + nntrees


def spenumerate(D, n, nz, nlogp, budget, paths, maxdepth=3):
    if budget < 0 or isterminal(n):
        yield nlogp, n
        return

    sources = paths[nz]

    for logp, args in sgroom(D, sources, nlogp, budget + nlogp, paths, maxdepth-1):
        n.tails = args
        yield logp, deepcopy(n)

def marknodes(D, Q, tree):
    z = 0
    paths = []
    qq = [tree]

    while len(qq) > 0:
        n = qq.pop(0)

        if not n.tails:
            paths.append([[]] * 2)
        else:
            sources = []
            for tail in n.tails:
                z += 1

                # idx tells for the index in D,
                # (q, z) for p of going and z where to
                # -1 means no entry for z
                dtails = [(-np.inf, -1)] * len(D)

                if not D.index(tail) is None:
                    dtails[tail.idx] = (Q[D.index(tail)], z)
                else:
                    print(f'big mistake - {tail}:{tail.type} is not in {D}')

                # bonus for the hole
                arrowidx = D.index(Delta('<>', ishole=True, type=tail.type))
                dtails[arrowidx] = (0, -1)

                sources.append(dtails)

                qq.append(tail)

            paths.append(sources)

    return paths

def count_ghosts(tree, ghost):
    if isequal(tree, ghost):
        return 1

    if not tree.tails:
        return 0

    out = 0
    for tail in tree.tails:
        out += count_ghosts(tail, ghost)

    return out


def chill_count(tree, ghosts):
    count = Counter()
    qq = [tree]

    while len(qq) > 0:
        n = qq.pop(0)

        for ghost in ghosts:
            if isequal(n, ghost):
                count[ghost] += 1

        if not n.tails: continue
        for tail in n.tails:
            qq.append(tail)

    return count


def count_simply(trees, ghosts):
    count = Counter()

    for ghost in ghosts:
        for tree in trees:
            count[ghost] += count_ghosts(tree, ghost)

    return count


def count_jive(D, Q, alltrees, trees):
    count = Counter()

    for tree in trees:
        for _, ghost in spenumerate(D, D[D.index(tree)], 0, 0, np.inf, marknodes(D, Q, tree), np.inf):
            c = 0
            for tree in alltrees:
                c += count_ghosts(tree, ghost)

            count[ghost] = c

    return count


def ghostsout(D, Q, trees):
    ghosts = set()
    for tree in trees:
        for _, ghost in spenumerate(D, D[D.index(tree)], 0, 0, np.inf, marknodes(D, Q, tree), np.inf):
            ghosts.add(ghost)

    return ghosts


def split(ncores, xs):
    l = len(xs) // ncores

    splitted = []
    for i in range(ncores+1):
        splitted.append(xs[i*l:min(len(xs),(i+1)*l)])
    splitted[-2].extend(splitted[-1])
    splitted.pop(-1)

    return splitted


def saturate(D, sols):
    ghosttime = time()
    trees = [normalize(s) for s in sols.values() if s]

    D.reset()

    print(f"size of the forest: {len(pickle.dumps(trees)) >> 10}M")

    while True:
        types = reduce(lambda acc, x: acc | x, [showoff_types(tree) for tree in trees])

        for tp in types:
            D.add(Delta('<>', ishole=True, type=tp))

        Q = th.log_softmax(th.ones(len(D)), -1)

        stime = time()

        splitted_trees = split(ncores, trees)

        if ncores > 1:
            try:
                pool = mp.Pool(ncores)
                counts = pool.starmap(count_jive, zip(repeat(D), repeat(Q), repeat(trees), splitted_trees))
            finally:
                pool.close()
                pool.join()

            counts = sum(counts, Counter())
        else:
            counts = count_jive(D, Q, trees, trees)

        print(f'counted those fellows in {(time() - stime) / 60:.2f}m')

        mx = sum(map(length, trees))

        mk = 0.99
        hiddentail = None

        for ghost, c in counts.items():
            nargs = 1 + countholes(ghost)

            mxj = mx - c * (length(ghost) - nargs)
            mj = length(ghost)

            k = (mxj + mj) / mx
            if k < mk:
                mk = k
                hiddentail = deepcopy(ghost)

        for dhole in D[D.index('<>'):]:
            D.pop(dhole)

        if hiddentail == None:
            print(f'ghosting took {(time() - ghosttime)/60:.2f}m')
            return trees

        tailtypes = typize(hiddentail)

        if len(tailtypes) == 0:
            name = hiddentail()
            df = Delta(name, type=hiddentail.type, hiddentail=hiddentail, repr=f"'{name}'")
        else:
            name = f"f{len(D.invented)}"
            df = Delta(name, type=hiddentail.type, tailtypes=tailtypes, hiddentail=hiddentail, repr=name)

        print(f"adding {df}: {df.type} with {df.hiddentail} #{mk:.3f}")

        trees = [replace(tree, df.hiddentail, df) for tree in trees]

        for tree in trees:
            freeze(tree)

        freeze(df)
        D.add(df)

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

def solve_needle(X, D, Q, solutions=None, maxdepth=10, ntries=100_000):
    print(f'{len(D)=}')

    if solutions is None:
        solutions = {x: None for x in X}

    cnt = 0
    stime = time()
    notsolved = sum([s is None for s in solutions.values()])

    paths, paths_terminal = makepaths(D, Q)
    requested_type = type(X[0])

    ephermal = Delta(None, None, tailtypes=[requested_type])
    D.add(ephermal)

    for wrapper in penumerate(D, ephermal, 0, 10, *makepaths(D, Q), maxdepth=maxdepth+1):
        tree = wrapper.tails[0]
        out = tree()

    while True:
        tree = newtree(D, requested_type, paths, paths_terminal, q=Q)
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


def solve_enumeration(X, D, Q, solutions=None, maxdepth=10, timeout=60, budget=0):
    print(f'{len(D)=}')

    tosolve = count_everyone(X)
    cnt = 0
    stime = time()

    requested_type = type(X)
    print(f"{requested_type=}")

    LOGPGAP = 2
    done = False

    def cb(tree, logp):
        nonlocal cnt, done, stime

        try:
            out = tree()
        except Exception as e:
            print(f"it's just a little mistake: {e} with {tree}")
            return

        if out == X:
            done = True

        cnt += 1

        if not(cnt % 100000) and cnt > 0:
            print(f'! {cnt/(time()-stime):.2f}/s')

            if time() - stime > timeout:
                done = True

        if out in X:
            if not out in solutions:
                print(f'[{cnt:6d}] caught {out} with {tree}')

            if not out in solutions or length(tree) < length(solutions[out]):
                solutions[out] = deepcopy(tree)

    if budget == 0:
        idx = 0
        while not done:
            cenumerate(D, Q, requested_type, (LOGPGAP * idx, LOGPGAP * (idx+1)), maxdepth, cb)
            idx += 1
    else:
        ephermal = Delta('root', ishole=True, tailtypes=[requested_type])
        D.add(ephermal)
        Q = th.hstack((Q, tensor([0])))

        for logp, wrapper in penumerate(D, ephermal, 0, budget, makepaths(D, Q), maxdepth=maxdepth+1):
            tree = wrapper.tails[0]
            cb(tree, logp)

        D.pop(ephermal)

    took = time() - stime
    print(f'total: {cnt}, took: {took/60:.1f}m, iter: {cnt/took:.0f}/s')
    print(f'solved: {sum(s is not None for s in solutions.values())}/{tosolve}')
    return solutions


def kcompress(D, trees):
    while True:
        ds = flatten(list(map(lambda tree: list(showoff_kids(tree)), trees)))
        count = Counter(ds)
        most_common = count.most_common()

        totall = sum(map(length, trees))
        mink = totall
        nd = None

        for d, c in most_common:
            if c < 3:
                continue

            d = tr(D, d)

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


def truly_largest_substring(string):
    for s1idx in range(len(string)):
        if string[s1idx] != '(':
            continue

        for e1idx in range(s1idx+2, len(string)):
            for s2idx in range(s1idx+1, len(string)):
                if string[s2idx] != '(':
                    continue

                for e2idx in range(s2idx+2, len(string)):
                    if BREAK in string[s1idx:e1idx]:
                        continue

                    if string[s1idx:e1idx] == string[s2idx:e2idx]:
                        yield string[s1idx:e1idx]

def findwrap(s, start):
    i = start
    nbrackets = 0
    while i < len(s):
        if s[i] == '(':
            nbrackets += 1

        if s[i] == ')':
            nbrackets -= 1

        if nbrackets == 0:
            return i

        i += 1

BREAK = " @ "

def getit(D, string, prefix):
    if prefix[-1] != ')':
        prefix = prefix[:-prefix[::-1].find(' ')-1]

    sidx = string.find(prefix)
    idx = sidx

    holeidx = 0
    nbrackets = 0
    mut = prefix
    pastprefix = False

    tailtypes = []

    while idx < len(string):
        if pastprefix and string[idx] not in "() ":
            se_idx = idx
            idx += 1

            while string[idx].isalnum() or string[idx] == "'":
                idx += 1

            expr = string[se_idx:idx]
            mut += f' ${holeidx}'
            holeidx += 1

            tailtypes.append(D[expr].type)

        if string[idx] == BREAK:
            break

        if string[idx] == ')':
            if pastprefix:
                mut += ')'

            nbrackets -= 1

        if string[idx] == '(':
            if pastprefix:
                ending = findwrap(string, ast)
                idx = tr(D, string[idx:ending+1])
                tailtypes.append(ast.type)

                mut += f' ${holeidx}'
                holeidx += 1
                idx = ending

            else:
                nbrackets += 1

        if nbrackets == 0:
            break

        if idx >= sidx + len(prefix):
            pastprefix = True

        idx += 1

    hiddentail = tr(D, mut)

    if len(tailtypes) == 0:
        name = hiddentail()
        df = Delta(name, type=hiddentail.type, hiddentail=hiddentail)

    else:
        name = f"f{len(D.invented)}"
        df = Delta(name, type=hiddentail.type, tailtypes=tailtypes, hiddentail=hiddentail, repr=f"{name} ({' '.join([f'${i}' for i in range(len(tailtypes))])}) {hiddentail}")

    return df

def count_occ(string, s):
    c = 0
    l = len(s)
    for sidx in range(len(string)-l+1):
        if string[sidx:sidx+l] == s:
            c += 1

    return c


def seesvd(D, mx, string, s):
    try:
        nd = getit(D, string, s)
    except:
        return [np.inf]

    c = count_occ(string, s)

    _repr = len(s.split(' '))
    if nd.tailtypes:
        nrepr = 1 + len(nd.tailtypes)
    else:
        nrepr = 1

    mxj = mx - c * (_repr - nrepr)
    mj = length(nd.hiddentail)

    k = (mxj + mj) / mx

    return k, c, nd


def count_everyone(X):
    subs = set()

    for l in range(1, len(X)+1):
        for sidx in range(len(X) - l+1):
            subs.add(X[sidx:sidx+l])

    return len(subs)


def ECD(X, D, timeout=60, budget=20):
    D.reset()

    Q = F.log_softmax(th.ones(len(D)), -1)

    tosolve = count_everyone(X)
    idx = 0
    sols = {}
    solved = False
    while not solved:
        sols = solve_enumeration(X, D, Q, sols, maxdepth=10, timeout=timeout, budget=budget + 2 * idx)

        trees = saturate(D, sols)
        idx += 1

        if X in sols:
            break

        Qmodel = dream(D, trees)

        ntoks = 100
        shift = max(randint(max(len(X) - ntoks, 1)), 0)
        Q = Qmodel(tc(X)[shift:shift+ntoks][None]).flatten().detach()
        Q = F.log_softmax(Q, -1)

    return {x: tree for x, tree in zip(sols, trees)}

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
    device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    ntoks = 100
    qmodel = RecognitionModel(len(D)).to(device)
    opt = th.optim.Adam(qmodel.parameters())
    paths, paths_terminal = makepaths(D, th.ones(len(D)))

    tbar = trange(100)
    for _ in tbar:
        trees = [newtree(D, str, paths, paths_terminal, depth=10) for _ in range(4)]
        for i in randint(len(soltrees), size=4):
            trees.append(soltrees[i])

        Xy = [[tree()[:ntoks], alld(tree)] for tree in trees]

        # x, i: d
        Xy = [[(xy[0], D.index(d)) for d in xy[1]] for xy in Xy]
        Xy = reduce(lambda acc, xy: acc + xy, Xy, [])

        X = [tc(xy[0])[None].to(device) for xy in Xy]

        q = th.vstack([qmodel(x) for x in X])
        y = tensor([xy[1] for xy in Xy], device=device)

        opt.zero_grad()
        loss = F.cross_entropy(q, y)
        loss.backward()
        opt.step()

        tbar.set_description(f'{loss=:.2f} batchsize={len(X)}')

    return qmodel.to('cpu')

if __name__ == '__main__':
    D = Deltas([
        Delta(add, int, [int, int], repr='+'),
        Delta(mul, str, [str, int], repr='*'),
        Delta(add, str, [str, str], repr='u'),
        Delta(2, int),
        Delta(3, int),
        Delta('0', str),
        Delta('1', str),
    ])

    X = "10001000100010001000"
    Z = ECD(X, D, budget=16)

    print(Z[X])
