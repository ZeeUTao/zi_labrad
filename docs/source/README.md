# README

python script to control Zurich instruments by using labrad.



## Requirements

- python3, (suggested: 3.6.4)

- [scalabrad](https://github.com/ZeeUTao/scalabrad)

  Binary distribution of scalabrad are distributed via [binary](https://bintray.com/labrad/generic/scalabrad#files)

  [source code of scalabrad](https://github.com/ZeeUTao/scalabrad).

- [pylabrad](https://github.com/ZeeUTao/pylabrad-zeeu)

  ```CMD
  pip install pylabrad
  ```

  or see the forked github repository: https://github.com/ZeeUTao/pylabrad-zeeu 

- zhinst from [Zurich instrument](https://www.zhinst.com/)





## Get to start

Change the directory in `\server\xxx.ini`, see `\server\README.txt`. 

You can map a directory as M disk `M:\` for the data storage and avoid changing the directory. 

```CMD
@echo off
subst m: /d
subst m: %cd%
```



- run [scalabrad-binary](https://bintray.com/labrad/generic/scalabrad#files) via `server\open_scalabrad`: 

  ```CMD
  M:\scalabrad\bin\labrad --registry file:///M:/Registry?format=delphi
  ```

  or other ways

  

- run `LabRAD.exe`, `Registry.exe`, `Grapher.exe` in `\server\`

- run `pyle_test.py` to test. 

  you can see the live view data in `Grapher.exe` if success. 



scripts for Zurich instrument are under review

- start ipython, and run BatchRun
- run the commands in `mp.py`, for example, mp.s21_scan



## Coding style

- S.I. units are used for the number without specified units (float, int...)





## Scripts

```python
conf 
"""Script to initiate instrument, which is rarely changed and used as a single '.py' file to avoid re-initiate instrument, when we reload the other frequently changed codes, like zurichHelper.py. """

mp 
"""multiplexed codes for running experiments """

BatchRun 
"""some commands for our daily running.
We always open an 'ipython', and type 'run BatchRun' to start;
Then use 'mp.s21_scan' or others 'mp.xxx' to implement different experiments."""

zurichHelper
"""interface for instruments especially for zurich"""

waveforms
"""codes for waveforms"""

adjuster
"""interface with sliders and buttons to adjust parameters"""

pyle_test
"""tested codes"""

```

