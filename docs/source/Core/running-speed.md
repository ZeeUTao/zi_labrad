# Running speed





## Loop

When use looping in python, a good way is using the generator in standard packages, like `range`, which is implemented in C. 

For example, we test in laptop

```python
time0=time.time()
for _ in np.arange(1e7): pass
print(time.time()-time0) 
# ~1.04
```

```python
time0=time.time()
i = 0
while i < 1e7: 
    i += 1
print(time.time()-time0) 
# ~1.34
```

```python
time0=time.time()
aa = range(int(1e7))
for _ in aa:
    pass
print(time.time()-time0)
# 0.39 
```



The second case is slowest from its assignment operation, which create a lot of new object since `int` is immutable, we can test it as

```python
time0=time.time()
i = 0
while i < 1e7: 
    i += 0.25
    i += 0.25
    i += 0.25
    i += 0.25
print(time.time()-time0)
# ~ 2.43
```



even in a mutable objects, it can also create new object when you operate them

```python
a = [1,2,3]
id0 = id(a[0])
a[0] += 1
id1 = id(a[0])
# (id0 == id1) >> False
```





### Create initial array

if we want to create [0,0,....0], or filling the array/list, using NumPy is a good choice

```
### from top to bottom cost time increase
[0]*(int(1e7))
np.zeros(int(1e7)) # numpy as np
```

Therefore, in the looping, 

- try to decrease the assignment steps

- Performance: 

  standard function written by C  > NumPy > list operation

- Do not frequently use type transform, like: `list(np.array([1,2,3...]))`

  

  

### TODO

