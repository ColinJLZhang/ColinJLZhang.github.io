---
layout:     post
title:      sublime text markdown 404
subtitle:   让人头痛的404
date:       2019-12-12
author:     Colin
header-img: img/post-bg-coffee.jpeg
catalog: true
comments: true
tags:
    - markdown
    - 工具
---
# 如题，解决步骤如下：
* Omnimarkup preview doesn't work
* rm -rf Packages/python-markdown
* Restart sublime = preview works!
* Restart sublime again = preview doesn't work anymore
* repeate from step 1

来源[url](https://github.com/timonwong/OmniMarkupPreviewer/issues/50).

但是又会出现新的问题：
“1 missing dependency was just installed.”

解决方案：

```
Sublime Text 3总是出现“1 missing dependency was just installed.”

原因及结果
https://github.com/wbond/package_control/issues/989

mac
1) 关闭ST，删除：
0_package_control_loader.sublime-package
0_package_control_loader.sublime-package-new
$ rm -f ~/Library/Application\ Support/Sublime\ Text\ 3/Installed\ Packages/0_package_control_loader.sublime-package*
启动ST，一会儿会收到要求重启的错误，再重启一次ST，如果仍会出现要求重启，请进行下一步。

2) 关闭ST，删除 0_package_control_loader.sublime-package* 和 bz2
$ rm -f ~/Library/Application\ Support/Sublime\ Text\ 3/Installed\ Packages/0_package_control_loader.sublime-package*
$ rm -rf ~/Library/Application\ Support/Sublime\ Text\ 3/Packages/bz2
启动ST，一会儿会收到要求重启的错误，再重启一次ST，应该不会再接收到提示了。

其他系统的命令：
windows
C:\> del /f "%APPDATA%\Sublime Text 3\Installed Packages\0_package_control_loader.sublime-package*"
C:\> del /f /s /q "%APPDATA%\Sublime Text 3\Packages\bz2"
或者
C:\> def /f "安装位置\Data\Installed Packages\0_package_control_loader.sublime-package*"
C:\> def /f /s /q "安装位置\Data\Packages\bz2"


linux
$rm -f ~/.config/sublime-text-3/Installed\ Packages/0_package_control_loader.sublime-package*
$rm -rf ~/.config/sublime-text-3/Packages/bz2
```

来源[url](https://www.cnblogs.com/Bob-wei/p/5712126.html)
**HAPPY HACKING!**