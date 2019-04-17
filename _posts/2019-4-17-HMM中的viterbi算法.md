---
layout:     post
title:      隐含马尔科夫链的viterbi算法的理解
subtitle:   viterbi算法的本质
date:       2019-04-17
author:     Colin
header-img: img/post-bg-ios9-web.jpg
catalog: true
comments: true
tags:
    - HMM
    - 概率论
    - viterbi
    - 路径规划
    - 动态规划
---

这篇博客主要是对viterbi算法的一个理解，对于维特比算法的基础可以参考[维基百科](https://zh.wikipedia.org/wiki/%E7%BB%B4%E7%89%B9%E6%AF%94%E7%AE%97%E6%B3%95)的讲述（[这里也有一篇不错的博客在说viterbi](https://www.cnblogs.com/pinard/p/6991852.html)）。

**那么本文的问题是viterbi算法在求解HMM隐藏链的好处在哪？**

首先我们来看一个例子：


我在北京的女朋友每天根据天气 {rainy，sunny} 决定当天的活动 {walk，shop，clean} 中的一种，她希望我能根据她的活动来猜测北京的天气。我只能从她的朋友圈看到她每天的活动，也就是我每天只能知道她今天是去散步，购物还是打扫房间了。这个例子其中 {rainy，sunny} 便是隐含状态链； {walk，shop，clean} 是可观测状态链；隐含状态（天气）之间的相互转换概率叫做状态转移概率；我女朋友每天干什么并没有直接的联系，但是她每天做什么却受到天气的影响，比如她比较喜欢下雨天打扫房间所以下雨天她打扫房间的概率是0.5，而她讨厌下雨天出去逛街，所以她下雨天出去逛街的概率是0.1，这个概率就是发射概率；还有个初始概率即最开始是晴天还是雨天的概率。
<br>
<center>
    ![imgasdaddas](..\img\article\viterbi1.webp)
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">图1 隐含状态与观测状态</div>
</center>
现在的问题就是女朋友过去三天的朋友圈显示她的状态是： {walk shop clean} 
<br>
北京这几天最可能的天气是什么？
<br><br>
##### 暴力求解
最简单的方法当然是我可以把北京这三天所有的天气状况列举出来，求出每一个天气的序列的概率就行了。<br>
比如北京的天气可能是：<br>
**S > S > S**（连续三个大晴天，它的概率是 0.4 * 0.6 * 0.6 * 0.3 * 0.6 * 0.1 = 2.592e-3)

所有的可能数量一共是2 * 2 * 2 = 8个序列。

##### viterbi算法
<br>
viterbi算法的思想在于计算到目前为止的最优路径，比如：
<br>第一天我女朋友去逛街了(walk)，
<br>第一天是晴天的概率 = 开始是晴天的概率 * 晴天逛街的概率（ 0.24 = 0.4 * 0.6 ）
<br>第一天是雨天的概率 = 开始是雨天的概率 * 雨天逛街的概率（ 0.04 = 0.4 * 0.1 ）
<br><br>**所以第一天大概率是晴天？ 我们目前还不能这么确定！** 
<br>**为什么这么说？ 如果我们只知道第一天的观察状态那么我们可以这样下结论但是我们现在已经知道三天的观测状态，我们拥有了更多的信息，我们可以运用第二天的信息来进一步的验证第一天的选择因为第二天的天气（隐含状态）是受到第一天天气的影响等我们计算第二天的隐含状态概率就会更加的明显了**
<br><br>第二天我女朋友去购物了(shop)，
<br>假如第一天是晴天，第二天是晴天的概率 = 第一天是晴天 * 第二天是晴天 * 晴天逛街的概率 （ 0.0432 = 0.24 * 0.6 * 0.3 ）
<br>假如第一天是雨天，第二天是晴天的概率 = 第一天是雨天 * 第二天是晴天 * 晴天逛街的概率 （ 0.0054 = 0.06 * 0.3 * 0.3 ）

<br>假如第一天是晴天，第二天是雨天的概率 = 第一天是晴天 * 第二天是雨天 * 雨天逛街的概率 （ 0.038 = 0.24 * 0.4 * 0.4 ）
<br>假如第一天是雨天，第二天是雨天的概率 = 第一天是雨天 * 第二天是雨天 * 雨天逛街的概率 （ 0.0168 = 0.06 * 0.7 * 0.4 ）

<br>**现在我们可以很安全的说第一天大概率是晴天！ 因为第二天无论是晴天还是雨天得出的结论总是第一天是晴天的概率最大。另一方面，我可以得出第二天是晴天的最大概率为0.432，是雨天的概率为0.038，好像第二天是晴天的概率大一点？我们看第三天的情况**

<br>同理我们可以计算第三天,
<br>假如第二天是晴天，第三天是晴天的概率 = 2.592e-3
<br>假如第二天是雨天，第三天是晴天的概率 = 1.152e-3

<br>假如第二天是晴天，第三天是雨天的概率 = 8.46e-3
<br>假如第二天是雨天，第三天是雨天的概率 = 13.4e-3（*）

<br>**第二天是雨天！ 当第二天是雨天第三天也是雨天的时概率最大**
<br>**所以最有可能的天气序列是 S->R->R(晴天->雨天->雨天)**

现在我们比较暴力方法和vibiter，实际上vibite利用了先验概率和后验概率，比如第一天的天气确定，首先我们可以通过初始概率得到第一天是晴天的概率，同时通过我们第二天的观测值和转移概率以及发射概率我们可以得到第一天第二天整个序列的概率最大值这时我们就确定了第一天的概率，注意在HMM中第一天的状态只会直接影响到第二天的状态，不会影响到第三天，所以我们可以在此时完全确定第一天的状态，于是在进行后续的概率计算时我们不再需要讨论第一天是雨天的情况了，viterbi在每一步都可以决策到前一天的最优选择，这样就极大的减少了不必要的计算。
<br>另外再提一句，整个过程实际上一个极大似然估计的过程。

下面放一张维基百科的动图，虽然不是这个例子但是很形象的说明了路径规划的优点。
<center>
    ![]( ..\img\article\Viterbi_animated_demo.gif)
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">
        图2 路径选择
    </div>
</center>

python代码部分
```python
# 打印路径概率表
def print_dptable(V):
    print("    ", end=' ')
    for i in range(len(V)): 
        print("%7d" % i, end=' ')
    print('\n')
 
    for y in V[0].keys():
        print("%.5s: " % y, end=' ')
        for t in range(len(V)):
            print("%.7s" % ("%f" % V[t][y]), end=' ')
        print('\n')
 
 
def viterbi(obs, states, start_p, trans_p, emit_p):
    """
    :param obs:观测序列
    :param states:隐状态
    :param start_p:初始概率（隐状态）
    :param trans_p:转移概率（隐状态）
    :param emit_p: 发射概率（隐状态表现为显状态的概率）
    :return:
    """
    # 路径概率表 V[时间][隐状态] = 概率
    V = [{}]
    # 一个中间变量，代表当前状态是哪个隐状态
    path = {}
 
    # 初始化初始状态 (t == 0)
    for y in states:
        V[0][y] = start_p[y] * emit_p[y][obs[0]]
        path[y] = [y]
 
    # 对 t > 0 跑一遍维特比算法
    for t in range(1, len(obs)):
        V.append({})
        newpath = {}
 
        for y in states:
            # 概率 隐状态 =    前状态是y0的概率 * y0转移到y的概率 * y表现为当前状态的概率
            (prob, state) = max([(V[t - 1][y0] * trans_p[y0][y] * emit_p[y][obs[t]], y0) for y0 in states])
            # 记录最大概率
            V[t][y] = prob
            # 记录路径
            newpath[y] = path[state] + [y]
 
        # 不需要保留旧路径
        path = newpath
 
    print_dptable(V)
    (prob, state) = max([(V[len(obs) - 1][y], y) for y in states])
    return (prob, path[state])
 
 
def main():
    states = ('Rainy', 'Sunny')
    observations = ('walk', 'shop', 'clean')
    start_probability = {'Rainy': 0.6, 'Sunny': 0.4}
    transition_probability = {
        'Rainy' : {'Rainy': 0.7, 'Sunny': 0.3},
        'Sunny' : {'Rainy': 0.4, 'Sunny': 0.6},
    }
    emission_probability = {
        'Rainy' : {'walk': 0.1, 'shop': 0.4, 'clean': 0.5},
        'Sunny' : {'walk': 0.6, 'shop': 0.3, 'clean': 0.1},
    }
    
    result = viterbi(observations,
                   states,
                   start_probability,
                   transition_probability,
                   emission_probability)
    return result
 
 
result = main()
print(result)
```
<br>输出
```
           0       1       2 

Sunny:  0.24000 0.04320 0.00259 

Rainy:  0.06000 0.03840 0.01344 

(0.01344, ['Sunny', 'Rainy', 'Rainy'])
```

**HAPPY CODING!**