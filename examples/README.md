# Setting

If you are the new user, the file `BatchRun_user.py` should be modified. 



First, you need specify the path of `zilabrad` repository in your computer

```
# 1. Change the path if you need
repository_zilabrad = r'M:\zi_labrad'
```


## generate default registry


```python
run BatchRun_user.py
```

If success, you can see `./user_example` and `./Servers/devices` in the registry editor

In Registry editor, you can copy the **user_example** into **your_name**

## Set your workspace

modify main function in `BatchRun_user.py`, as following


```python
if __name__ == '__main__':
    ss = connect_ZI(reset=False, user='your_name')
```

Copy the script that your want edit in this folder, usually `multiplex.py` ,
rename it as you like, and import it. 