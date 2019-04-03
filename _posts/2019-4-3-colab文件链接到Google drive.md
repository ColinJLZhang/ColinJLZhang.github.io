---
layout:     post
title:      colab文件关联到Google drive
subtitle:   使用SDK或者使用mount
date:       2019-04-03
author:     Colin
header-img: img/post-bg-ios9-web.jpg
catalog: true
comments: true
tags:
    - colabtory
    - gdrive
    - deep_learning
---

刚接触到Colab的时候觉得十分开心，可以在云端运行，方便的共享模式，还不用适配各种环境，简直不要太爽，还一点就是免费的GPU运行可还行？但是面临的一个问题就是Colab会为每一个笔记本生成一个VM，这个环境最多持续运行12小时，超出这个时间，对不起Google会关掉你的服务（动了挖矿心思的坏人，被查到会被禁止使用服务哦），然后当你重新连接的时候所有东西都被“一键还原”了，这相当于你结婚白天忙了一天晚上准备入洞房的时候告诉你新娘跑了一样.....除此之外，如果你不小心关掉了你的浏览器，遭遇了神秘力量的断网，那么结果the same.


其实上面的一个问题还不算是太头痛，一般情况下12小时还是可以做大部分的工作，而且保证尽量不要断开colab页面就行了。但是我们在运行某一个项目的时候总是需要大量的文件夹，如果我们每次一个一个的上传到这个VM，第一是麻烦，另一个就是发生了断开的情况那简直是欲哭无泪啊.....

所以，一个较好的解决办法就是使用colab与google drive关联. Colab做计算，gdirve做存储.我们可以把文件存储在gdrive，这样调用起来也方便。

## 1.使用drive.mount()

最好的当然是官方说的方法呀：
    
    from google.colab import drive
    drive.mount('/content/gdrive') #这里相当于把你的Google drive 挂载在gdrive下面

这里会要求授权相应的gdrive(当然你得先要有一个gdrive!),点击url授权->输入授权码，ok.

授权完成后我们只需要把我们的工作路径放到我们想要工作的文件夹下面就可以了：

    import os
    os.chdir("gdrive/My Drive/Colab Notebooks/NST") #换成你自己的工作文件夹
    #我在我的Google drive的Colab Notebooks文件夹下面创建了NST文件存放我这个项目的所有文件

这时候你可以点击左侧有一个文件系统，这样你就可以查看到你的文件是否是可以访问的了.现在你就可以愉快的玩弄Google提供给你的GPU了.：D

如果你确实无聊的话，也可以看看下面的SDK方法=_=,我一开始的时候没看官方文档在网上找到的各种大神解决方案不得不服.

## 2.使用Google drive SDK

废话少说直接上代码：

    !apt-get install -y -qq software-properties-common python-software-properties module-init-tools
    !add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
    !apt-get update -qq 2>&1 > /dev/null
    !apt-get -y install -qq google-drive-ocamlfuse fuse

    from google.colab import auth
    auth.authenticate_user()
    from oauth2client.client import GoogleCredentials
    creds = GoogleCredentials.get_application_default()
    import getpass
    !google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
    vcode = getpass.getpass()
    !echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

这里会要求授权相应的gdrive(当然你得先要有一个gdrive!),点击url授权->输入授权码，ok.

授权完成后我们还需要在系统下面创建一个文件夹供我们使用：

    !mkdir -p drive
    !google-drive-ocamlfuse drive

最后我们把工作路径切换到我们的目录下就可以开始愉快的玩耍了：

    import os
    os.chdir("drive/Colab Notebooks/NST")
    #这里的路径请输入你准备在gdrive工作的路径，我这里是在gdrive下新建了一个Colab Notebooks/NST的文件夹

这里你就完成了文件的目录关联，现在你可以像在本地工作一样来调用文件了。（当然你先要把你的项目文件上传到gdrive，早说啊喂！）你可以通过命令查看当前的路径：

    import os
    os.getcwd()

**HAPPY CODING!**