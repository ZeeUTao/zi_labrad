# Qubit Server



## runQubits

- make sequences (in local PC)
- setup microwave sources
- setup parameters of DAC ADC devices and upload
- run devices and get data



## RunAllExperiment

- prepare parameters
- `runQubits`
- send processed data in data vault
- repeat until abort or end



## Key Parameters

- `awgs_pulse_len` 

  the maximum length of AWG pulse. 

  Example: when you implement T1 measurement, 

  ```python
  for qb in qubits:
  	qb['awgs_pulse_len'] += np.max(delay)
  ```

  otherwise, the AWG (Zurich HD) will not prepare the pulse with the length exceeding our default maximum value (stored in Registry).



## Cost of time

Since more devices can operate simultaneously, *setup devices, upload setting and wait for data* will not be too long and only related to the ports of one devices (8 for Zurich HD).

For making sequences, time is around a+N t, where a is the cost time for compiling program almost fixed, 





We mainly want to optimize the part of making sequences, which is slow in *python* and related with your qubit number. 

One of the solutions is using numpy (always array but not list) and [numba](https://numba.pydata.org/), where the last one can be simply used by adding a decorator at the start of your function

```python
from numba import jit

@jit(nopython=True)
def yourFunction(*args):
```

Numba can largely speed up the calculation involved loop and numpy, when the data is too large. If your data is not too large, pure numpy is enough. 



Attention: 

- Not to use a pure python to generate a large list, but use numpy and numba to generate.

  Some expression can not be used, for example:

```python
(start<=t<end)
```

​	which should be replaced by 

```python
(start<=t)*(t<end)
```

​	to be processed by numpy. 







