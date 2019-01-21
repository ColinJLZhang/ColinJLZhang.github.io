---
layout:     post
title:      算法练习-googol数第K位数状态
subtitle:   python实现
date:       2019-1-17
author:     Colin
header-img: img/post-bg-debug.png
catalog: true
comments: true
tags:
    - 算法
    - python
    - CJK 习题
---
### 题目：

简短翻译版：  
有这样一个01字符串，其中reverse用来逆序字符串switch用来取反字符串中的01

S0 = ""

S1 = "0"

S2 = "001"

S3 = "0010011"

S4 = "001001100011011"

...

SN = SN-1 + "0" + switch(reverse(SN-1)).

求，当N = googol 时，输入整数K，输出SN的第K位.(注：googol=10^100)

___________
原文（©Google）：  

    Problem  

    A "0/1 string" is a string in which every character is either 0 or 1.

    There are two operations that can be performed on a 0/1 string:  

    switch: Every 0 becomes 1 and every 1 becomes 0. For example, "100" becomes "011".

    reverse: The string is reversed. For example, "100" becomes "001".
    
    Consider this infinite sequence of 0/1 strings:

    S0 = ""

    S1 = "0"

    S2 = "001"

    S3 = "0010011"

    S4 = "001001100011011"

    ...

    SN = SN-1 + "0" + switch(reverse(SN-1)).

    You need to figure out the Kth character of Sgoogol, where googol = 10^100.

    Input
    The first line of the input gives the number of test cases, T. Each of the next T lines contains a number K.

    Output
    For each test case, output one line containing "Case #x: y", 
    where x is the test case number (starting from 1) and y is the Kth character of Sgoogol.

    Limits
    1 ≤ T ≤ 100.
    Small dataset
    1 ≤ K ≤ 10^5.
    Large dataset
    1 ≤ K ≤ 10^18.
    Sample

    Input 

    Output 

    4  Case #1: 0
    1  Case #2: 0
    2  Case #3: 1
    3  Case #4: 0
    10

### 思路

最简单的做法当然是暴力计算，当然这里无法计算出整个Sgoogol序列，这样的话存储这个序列大概要上千个T，正确的做法是需要第几个就计算输出第几个并且不保存序列，这样计算有一个小技巧可以提高算法的效率，先将所有要计算位置进行排序，只要计算一次到最大的数就可以结束了，这种方法应该是可行的，只是效率低下，这里我们讨论另一种方法，通过找寻规律直接计算值。


将SN考虑为一个二叉树，其根节点为0，左子树为SN-1，右子树为switch(reverse(SN-1))[sr(SN-1)]
         
        SN:
           0
        /     \
    SN-1     sr(SN-1)  

实际上，sr函数只会改变二叉树的根节点内容，查看如下S4和rs(S4),他们只有根节点取反其子树完全一致，原因在于S4的右子树本身是对S4的左子树rs，所以S5的右子树为rs(S4_l)=S4_r。所以我们可以看到二叉树的每层都是0 1 交替。所以我们只要确定K在二叉树的哪一层的第几个元素就可以求出该元素的值。

![img](..\..\..\..\img\article\1.png)         <!--因为网页的二级域名是根据日期生成的所以必须把文件夹退回到顶级域名-->
可以看到无论是S3的第二个元素还是S4的第二个元素，他们都是同一个元素，这是因为后一个序列是在前一个序列的基本结构上增加的结果，所以我们在求一个序列中的某个元素时，我们可以先计算出它属于的最小序列，比如S100的第一个元素和S1的第一个元素是完全相同的所以我们不必考虑S100只用考虑S1就可以了。

假设要查找第K个元素，它在第F层第x个元素上，那么存在如下关系：

$(K-2^{F-1})/2^F = x$

我们在实现过程中先通过log(K+1)求得K属于哪一个最小的序列，然后计算这个序列最多有多少层然后在进行循环，当求得整数的时候可以算出属于第几层，然后求出x就可以实现了。

### 代码
下面是python的代码：

```python
t = int(input())  # read a line with a single integer
for i in range(1, t + 1):
    num = int(input())
    if num==1:
        res = 0
    else:
        for x in range(0,72):#x为层数
            start = 2**x#每层二叉树的开始元素
            m = start*2
            if num%m==start:
                #print(x,start,m)
                if int((num-start)//m+(num-start)%m+1)%2:
                    res = 0
                    break
                else:
                    res = 1
                    break
    print("Case #{}: {}".format(i,res))
```
下面是Google提供的测试数据，分别是大数据集和小数据集：

[大数据集](https://github.com/ColinJLZhang/ColinJLZhang.github.io/blob/master/files/B-large-practice.in)    
[小数据集](https://github.com/ColinJLZhang/ColinJLZhang.github.io/blob/master/files/B-large-practice.in)    
[test](file://..\..\..\..\files\B-small-practice.in)

