# zi_labrad

python script to control Zurich instruments by using labrad.



## Requirements

- python3 (suggested: 3.7)

- Clone the repository and

  ```bash
  pip install -r requirements.txt
  ```

  

- [scalabrad](https://github.com/ZeeUTao/scalabrad)

  Binary distribution of scalabrad are distributed via [binary](https://bintray.com/labrad/generic/scalabrad#files). Here is [source code of scalabrad](https://github.com/ZeeUTao/scalabrad). 

  To run the binary distribution

  > You'll need to have Java 8 installed on your system. 

  and add the environmental variable `JAVA_HOME` to be the directory of installed Java8, for example `D:\Java8`

  

## Documentation

We use sphinx for documentations, you can build the documentation locally.

Make sure that you have the extra dependencies required to install the docs

```bash
pip install -r docs_requirements.txt
```

Go to the directory `docs` and

```bash
make html
```

This generate a webpage, index.html, in `docs/_build/html` with the rendered html.