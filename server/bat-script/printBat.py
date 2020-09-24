import sys
import os

path = r'pyle'

path_py = []
for roots,dirs,files in os.walk(path):
    for name in files:
        ext = os.path.splitext(name)[1]
        if ext != '.py':
            continue
        dir1 = os.path.join(roots, name)
        path_py.append(dir1)

file = open('bat_py2to3.bat','w')

for p in path_py:
    file.write('2to3 -w '+str(p)+'\n')
file.close()