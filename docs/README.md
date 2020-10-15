## README

See the documentation of html via `\build\html\index.html`





You can clone the repository, and run

```CMD
.\make.bat html
```



### Create new doc

create new folder

```
HowToDocProjOfPy\
    docs\
    mymodule\
    README.md
```

install via

```CMD
pip install sphinx sphinx-autobuild sphinx_rtd_theme
pip install sphinxcontrib-napoleon
pip install sphinxcontrib-apidoc
pip install m2r2
```

initiate

```CMD
sphinx-quickstart
```

then CMD will show an interactive commands to build the project, like below,

```
Welcome to the Sphinx 3.2.1 quickstart utility.

Please enter values for the following settings (just press Enter to
accept a default value, if one is given in brackets).

Selected root path: .

You have two options for placing the build directory for Sphinx output.
Either, you use a directory "_build" within the root path, or you separate
"source" and "build" directories within the root path.
> Separate source and build directories (y/n) [n]: y
```

generate html

```CMD
.\make.bat html
```

you can see the html via `\build\html\index.html`



#### Style

We prefer to change the style defined in `docs\source\conf.py`, from 

```python
html_theme = 'alabaster'
```

to

```python
# html_theme = 'alabaster'
import sphinx_rtd_theme
html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
```



#### Support markdown

change `docs\source\conf.py` 

```python3
extensions = [
    'sphinxcontrib.napoleon',
    'sphinxcontrib.apidoc',
    'sphinx.ext.viewcode',
    'm2r2'
]
```

and 

```
source_suffix = ['.rst', '.md'] 
```



#### apidoc_module_dir

add the following code in`docs\source\conf.py` 

- Note: project_path is teh path of your python file

```python
import os
import sys
project_path = '../../mymodule'
sys.path.insert(0, os.path.abspath(project_path))

# ...

apidoc_module_dir = project_path
apidoc_output_dir = 'python_apis'
# apidoc_excluded_paths = ['tests']
apidoc_separate_modules = True
```

