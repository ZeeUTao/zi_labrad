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



## Time order

- DC bias start previously
- XY, Z pulses
- readout
- DC bias end

Some important parameters are given below

```python
# int: sample number for one sweep point
qa.set_result_samples(q_ref['stats'])

# only related with wiring and devices, delay between QA
# signal output and demodulation
qa.set_readout_delay(q_ref['readout_delay'])

# set qa pulse length in AWGs, and set same length for demodulate.
qa.set_pulse_length(q_ref['readout_len'])

# delay between zurich HD and QA
qa.set_adc_trig_delay(q_ref['bias_start']['s']+q_ref['experiment_length'])

```



## Caution

- If qubits different `xy_mw_fc`, you need specify the sideband frequency correctly. 

  For example, the microwave source only send the carrier frequency according to the first qubit, then all of the other qubit should has

  ```python
   f_sideband =qubit['f10'] - qubit['xy_mw_fc']
  ```

  



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







