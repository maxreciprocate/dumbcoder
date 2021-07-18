## dumbed down version of karpathy/mingpt
import math
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch import tensor, as_tensor, from_numpy
from numpy.random import rand, randint, randn, normal


class Attend(nn.Module):
    def __init__(self, nheads, nembd, ntoks, pdrop=0.0):
        super().__init__()
        self.key = nn.Linear(nembd, nembd)
        self.query = nn.Linear(nembd, nembd)
        self.value = nn.Linear(nembd, nembd)

        self.attn_drop = nn.Dropout(pdrop)
        self.head_drop = nn.Dropout(pdrop)
        self.head = nn.Linear(nembd, nembd)
        self.nheads = nheads

        self.register_buffer("mask", th.tril(th.ones(1, 1, ntoks, ntoks)) == 0)

    def forward(self, x):
        nbatch, ntoks, nembd = x.size()

        # (nbatch, nheads, ntoks, nembd)
        k = self.key(x).view(nbatch, ntoks, self.nheads, nembd // self.nheads).transpose(1, 2)
        q = self.query(x).view(nbatch, ntoks, self.nheads, nembd // self.nheads).transpose(1, 2)
        v = self.value(x).view(nbatch, ntoks, self.nheads, nembd // self.nheads).transpose(1, 2)

        att = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))
        att = att.masked_fill(self.mask[:, :, :ntoks, :ntoks], -np.inf)
        att = F.softmax(att, -1)
        att = self.attn_drop(att)

        out = att @ v
        out = out.transpose(1, 2).contiguous().view(nbatch, ntoks, nembd)

        return self.head_drop(self.head(out))


class AttendClose(nn.Module):
    def __init__(self, nheads, nembd, ntoks, pdrop=0.1):
        super().__init__()
        self.key = nn.Linear(nembd, nembd)
        self.query = nn.Linear(nembd, nembd)
        self.value = nn.Linear(nembd, nembd)
        self.register_buffer("mask", th.tril(th.ones(ntoks, ntoks)) == 0)
        self.attention = nn.MultiheadAttention(nembd, nheads, dropout=pdrop, bias=False)
        self.head = nn.Linear(nembd, nembd)
        self.head_drop = nn.Dropout(pdrop)

    def forward(self, x):
        ntoks, nbatch, nembd = x.size()

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        out, _ = self.attention(q, k, v, attn_mask=self.mask[:ntoks, :ntoks])

        return self.head_drop(self.head(out))


class Block(nn.Module):
    def __init__(self, nheads, nembd, ntoks, pdrop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(nembd)
        self.ln2 = nn.LayerNorm(nembd)

        self.att = Attend(nheads=nheads, nembd=nembd, ntoks=ntoks)
        # self.att = AttendClose(nheads=nheads, nembd=nembd, ntoks=ntoks)

        self.head = nn.Sequential(
            nn.Linear(nembd, 4 * nembd),
            nn.GELU(),
            nn.Linear(4 * nembd, nembd),
            nn.Dropout(pdrop)
        )

    def forward(self, x):
        x = x + self.att(self.ln1(x))
        x = x + self.head(self.ln2(x))

        return x

class YOGPT(nn.Module):
    def __init__(self, vocabsize, nheads, nembd, ntoks, nlayers, pdrop=0.0):
        super().__init__()

        self.vocabsize = vocabsize
        self.nheads = nheads
        self.ntoks = ntoks
        self.nembd = nembd

        self.tok_emb = nn.Embedding(self.vocabsize, self.nembd)
        self.pos_emb = nn.Parameter(th.zeros(1, self.ntoks, self.nembd))
        self.drop = nn.Dropout(pdrop)

        self.blocks = nn.Sequential(*[Block(nheads=nheads, nembd=nembd, ntoks=ntoks, pdrop=pdrop) for _ in range(nlayers)])
        self.ln = nn.LayerNorm(nembd)
        self.head = nn.Linear(nembd, vocabsize, bias=False)

        print(f'{sum(p.numel() for p in self.parameters()) / 2**20:.2f}M params')

    def forward(self, x):
        bsize, ntoks = x.size()

        embs = self.tok_emb(x)
        positions = self.pos_emb[:, :ntoks, :]

        x = self.drop(embs + positions)
        out = self.blocks(x)
        # logits = self.head(self.ln(out))
        logits = self.ln(out)

        return logits

def gsample(model, xs, ngrow):
    model.eval()

    for _ in range(ngrow):
        logits = model(xs)
        conts = logits[:, -1, :].argmax(-1).unsqueeze(-1)

        xs = th.cat((xs, conts), dim=1)

    return xs

def pprint(d: dict):
    for k,v in d.items():
        print(f'{k}: {v}')

def sample(ps):
    ps /= ps.sum()
    cdf = ps.cumsum(-1)
    x = rand()
    for i in range(len(ps)):
        if cdf[i] > x:
            return i

    return len(ps)-1
