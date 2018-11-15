---
layout:     post
comments:   true
title:      文献阅读
subtitle:   3d视频描述算子
date:       2018-11-14
author:     Colin
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags: 
    - video analysis
    - computervision
    
---
1. Klaser A, Marszałek M, Schmid C. A spatio-temporal descriptor based on 3d-gradients[C]//BMVC 20OmniMarkupPreviewer08-19th British Machine Vision Conference. British Machine Vision Association, 2008: 275: 1-10.
    * **简介**：提出了一种3d的描述子描述视频特征描述子基于20面体的投影。将x,y,t在一个立方体中求平均梯度 $iv_{\partial_x}$, $iv_{\partial_y}$, $( iv_{\partial_t}$,把三个平均梯度投影到一个二十面体，一个二十面体可以这样表示：  
    \\[  
        (\pm1,\pm1,\pm1), \quad (0,\pm\frac{1}{\varphi},\pm\varphi),\quad (\pm\frac{1}{\varphi},\pm\varphi,0), \quad(\pm\varphi,0,\pm\frac{1}{\varphi})
        \quad where \quad \varphi = \frac{1+\sqrt{5}}{2}
    \\]  
    投影公式：  
    \\[  
        \hat{q}_b=(\hat{q}_{b1},...,\hat{q}_{bn})^T=\frac{P\cdot\bar{g}_b}{\begin{Vmatrix} \bar{g}_b\end{Vmatrix}}_2
    \\]  
    * **问题**： 
        +  为什么要把平均灰度投影到正二十面体的各个面上进行直方图统计？
        +  阈值化的理由不是很清楚，$t=p^T_i \cdot p_j \approx 1.29107$是如何计算的？
        +  SVM及参数学习是如何实现的？




   
---
**文献积累**  

+ 【orthogonal space of optiacl flow】 B. G. Horn, Berthold K.P.; Schunck. Determining Optical Flow. Artificial Intelligence 17:185–203, 1981.  
+ 【a fast video-level descriptor】 B. Fernando, E. Gavves, J. Oramas, A. Ghodrati, and T. Tuytelaars. Rank pooling for action recognition. T-PAMI,
39(4):773–787, 2017.

