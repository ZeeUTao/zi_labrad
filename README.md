# zi_labrad

python script to control Zurich instruments by using labrad.



## Requirements

- python3, (suggested: 3.6，3.7)

- [scalabrad](https://github.com/ZeeUTao/scalabrad)

  Binary distribution of scalabrad are distributed via [binary](https://bintray.com/labrad/generic/scalabrad#files). Here is [source code of scalabrad](https://github.com/ZeeUTao/scalabrad). 

  To run the binary distribution

  > You'll need to have Java 8 installed on your system. 

  and add the environmental variable `JAVA_HOME  ` to be the directory of installed Java8, for example `D:\Java8`

  

- [pylabrad](https://github.com/ZeeUTao/pylabrad-zeeu)

  ```bash
  pip install pylabrad
  ```

  or see the forked github repository: https://github.com/ZeeUTao/pylabrad-zeeu 

  

- zhinst from [Zurich instrument](https://www.zhinst.com/)



## Documentation

Install the following requirements for docs building via

```bash
pip install sphinx sphinx-autobuild sphinx_rtd_theme
pip install sphinxcontrib-napoleon
pip install sphinxcontrib-apidoc
pip install m2r2
```

Clone the repository, and run the following bash

```bash
.\docs\make html
```