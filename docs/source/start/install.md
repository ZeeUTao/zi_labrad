# 安装



## Python

下载 [anaconda](https://www.anaconda.com/download) ，选择Python版本为3.7 。



## scalabrad

安装scalabrad的一种方法是，点击链接进入github原仓库  [scalabrad](https://github.com/labrad/scalabrad)，或者我们创建的备份 [forked scalabrad](https://github.com/ZeeUTao/scalabrad)，然后参考他们的文档执行。

考虑到我们无需对其进行修改，开发，另一种更简单的方法是，直接进入 [binary](https://bintray.com/labrad/generic/scalabrad#files)，网站提供压缩包下载，比如
[scalabrad-0.8.3.tar.gz](https://bintray.com/labrad/generic/download_file?file_path=scalabrad-0.8.3.tar.gz)。下载完成后直接用解压缩软件解压，记住解压的文件目录。

最好选择一个专门的文件夹把映射成一个盘（比如M盘），具体的操作为：

- 进入文件夹的目录，
- 进入CMD命令行，在命令行里输入命令

```bash
@echo off
subst m: /d
subst m: %cd%
```

然后这个文件夹就可以当 M 盘使用了，我们把软件解压到那个文件夹中，以 M:\scalabrad-0.8.3 为例，
CMD内输入如下

```bash
M:\scalabrad-0.8.3\bin\labrad --registry file:///M:/Registry?format=delphi
```

其中 file:///后需要改成 Registry 的目录（一般实验存参数使用）



## pylabrad

我们需要用 python 调用 labrad，所以需要 [pylabrad](https://github.com/labrad/pylabrad)，直接通过 anaconda 或者 pip 安装

```bash
pip install pylabrad
```

我们在 github 也做了备份仓库，见 [pylabrad](https://github.com/ZeeUTao/pylabrad-zeeu)



## zhinst

参考 Zurich 官网的介绍：[zhinst](https://www.zhinst.com/)

