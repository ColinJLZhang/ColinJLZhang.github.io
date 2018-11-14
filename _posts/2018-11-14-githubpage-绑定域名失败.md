---
layout:     post
title:      Page 绑定域名失败
subtitle:   
date:       2018-11-14
author:     Colin
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags: 
    - 建站
    - 域名解析
---

#  “Domain's DNS record could not be retrieved”解决方法
如果你遇到上述问题，检查看看你的解析有没有如下三条设置，没有的话请加上：
     
@         A             192.30.252.153

@         A             192.30.252.154

www      CNAME           username.github.io.

