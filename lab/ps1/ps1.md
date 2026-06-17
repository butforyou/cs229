# ps1

1.problem1

![image-20260615230745732](E:\杂七杂八\typora,picture\image-20260615230745732.png)

-a.prove

对于逻辑回归,往往将其表征为概率的形式，常见形式为$h(x)=\frac{1}{1+e^{-\theta^{T} x}}$,$P(t)=\begin{cases} t &\text{if} \: \: t>0.5 \\1-t &\text{if}\: \:t<0.5 \end{cases}$

对于输入$x$有$[x_{1},x_{2}....x_{n}]^{T}$,权重$\theta$为$[\theta_{1},\theta_{2}...,\theta_{n}]^{T}$

那么对于梯度求导则有,$\frac{\partial \: h(x)}{\partial \: \theta_{i}}=h^{'}(x)x_{i}$,同时易知$h^{'}(x)=h(x)(1-h(x))$

那么对于$\theta$的梯度有$h^{'}(x)*[x_{1},x_{2},...x_{n}]^{T}$

对于已知的y可以总结为$P(y|x;\theta)=h_{\theta}(x)^{y}(1-h_{\theta}(x))^{1-y}$

使用最大似然估计方法进行概率累计乘积，同时进行取对数可得到如题的似然函数,进行训练参数更新往往是用梯度下降法,所以对于似然函数是取负:$J(\theta)=-\frac{1}{n} \displaystyle \sum_{i=1}^{n}y_{i}log(h_{\theta}(x_{i}))+(1-y_{i})log(1-h_{\theta}(x_{i}))$

对于似然函数进行求偏导可得:$$\frac{\partial J(\theta)}{\partial \theta}=- \displaystyle \frac{1}{n} \sum_{i=1}^{n} y_{i}\frac{h_{\theta}(x)(1-h_{\theta}(x))x}{h_{\theta}(x)}-(1-y_{i})\frac{h_{\theta}(x)(1-h_{\theta}(x))x}{1-h_{\theta}(x)}=-\frac{1}{n} \sum_{i=1}^{n}y_{i}(1-h_{\theta}(x_{i}))x_{i}-(1-y_{i})h_{\theta}(x_{i})x_{i}=-\frac{1}{n} \sum_{i=1}^{n}\{y_{i}-h_{\theta}(x_{i})\}x_{i}$$

常规梯度下降法为$\theta_{i+1}=\theta_{i}-\frac{\partial J(\theta)}{\partial\theta}$,易见这是对一整个向量进行梯度下降

但是本题要求使用牛顿法，求解黑森矩阵，二次下降更快。

在数值分析中学习到的牛顿法:$f(x^{*})=f(x_{i})+f'(x_{i})(x^{*}-x_{i})$,其中$f(x^{*})=0$得到$x_{i+1}=x_{i}-\frac{f(x_{i})}{f'(x_{i})}$

但是牛顿法是求$f(x)=0$的点，也就是我们一阶导所要的驻点。那么就有将$f(x)$带入$J(\theta)$,$\theta_{i+1}=\theta_{i}-\frac{J'(\theta)}{J''(\theta)}=x_{t+1} = \theta_t - \nabla^2 f(\theta_t)^{-1} \nabla f(\theta_t) \quad x_t \in \mathbb{R}^n$

对于一阶导已经求出是一个列向量形式，因此进行二阶导的推导:
$$
\nabla^{2}f(\theta)_{(e,f)}= \frac{1}{n}\sum_{i=1}^{n}x_{f}h_{\theta}(x_{i})(1-h_{\theta}(x_{i}))x_{e}
$$
那么黑森矩阵就可以表示为$H=X^{T}ZX$,$Z=h_{\theta}(x_{i})(1-h_{\theta}(x_{i}))\ge0$,因此半正定性成立
