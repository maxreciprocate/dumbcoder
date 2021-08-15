This is a dumbed down version of the [dreamcoder](https://github.com/ellisk42/ec)

For demonstration let's induce a simple program for some string X using the set of expressions D
```python
>>> X = '10001000100010001000'
>>> D = Deltas([
    Delta('0', str),
    Delta('1', str),
    Delta(2, int),
    Delta(3, int),
    Delta(add, int, [int, int], repr='+'), # 2 + 3 = 5
    Delta(mul, str, [str, int], repr='*'), # '0' * 2 = '00'
    Delta(add, str, [str, str], repr='u'), # '1' + '0' = '10'
])
>>> Z = ECD(X, D)
>>> Z[X]
(* '1000' (+ 2 3))
>>> Z[X]()
'10001000100010001000'
```

ECD extends the starting DSL D with new expressions, in this case it invented a '1000' since there are a lot of these in X

```python
>>> D['1000'].hiddentail
(u '1' (* '0' 3))
>>> D['1000']()
'1000'
```

ECD stands for explore, compress and dream - three stages of the algorithm: finding X by probabilistically enumerating expressions from DSL, compressing found representations and then training recognition model using those for the next enumeration cycle.
