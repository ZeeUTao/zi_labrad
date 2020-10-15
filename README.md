# zi_labrad

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



## Documentation

Install the following requirements for docs building via

```
pip install sphinx sphinx-autobuild sphinx_rtd_theme
pip install sphinxcontrib-napoleon
pip install sphinxcontrib-apidoc
pip install m2r2
```

Clone the repository, and run the following bash

```
.\docs\make html
```