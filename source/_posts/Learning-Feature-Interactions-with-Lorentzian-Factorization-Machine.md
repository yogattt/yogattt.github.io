---
title: Learning Feature Interactions with Lorentzian Factorization Machine
date: 2021-11-16 17:02:14
tags: 
    - AAAI
    - 2020
    - FM
    - CTR
categories: 推荐系统
mathjax: true
---
## 摘要

&emsp; &emsp; 学习特征交互的表示以模拟用户行为对于推荐系统和点击率（CTR）预测至关重要。深度学习方法能够学习复杂的特征交互,但是这些方法需要大量与low-level representations相结合的训练参数，因此内存和计算效率都很低。  

&emsp; &emsp;作者提出了一个名为“LorentzFM”的新模型，该模型可以学习嵌入在双曲空间（hyperbolic space）中的特征相互作用，在双曲空间中，Lorentz距离的三角不等式的破坏是可实现的。双曲三角形的特殊几何特性对学习特征交互的表示是有益的。并且因为不需要任何顶部深度学习层，参数数量显著减少（20%至80%）。  

&emsp;&emsp;作者提出在使用Lorentz距离的双曲空间中学习低维表示，这样就能违反特征向量的三角不等式。该想法是受collaborative metric learning (CML) (Hsieh et al. 2017) 工作的启发。不同的是，CML作者认为欧氏空间中的三角形不等式应该严格遵守，本文作者提出利用三角形不等式的符号。具体地说，本文作者不是通过特征向量之间的内积或距离，而是通过检查它们在双曲空间中形成的三角形来构造特征交互的分数函数。采取这种做法的原因有两个方面：  
&emsp;&emsp;&emsp;&emsp;1.双曲空间本质上比欧几里德空间更广阔；  
&emsp;&emsp;&emsp;&emsp;2.提出的分数函数将提供一个鲁棒的目标函数来学习细粒度特征交互。  

## 技术背景

### **双曲几何(Hyperbolic Geometry)**

[推荐阅读](https://www.cnblogs.com/baiting/p/11006331.html)  
&emsp;&emsp; 双曲几何旨在研究具有常数负曲率的非欧几里德空间。由于其负曲率，双曲几何与欧几里德几何相比具有非常不同的性质。  
&emsp;&emsp; 首先，与欧几里德空间中的线性和二次增长率相反，双曲空间中圆的周长和面积随半径呈指数增长。因此，在半径上界相同的双曲空间中嵌入的容量比欧几里德空间中嵌入的容量大得多。其次，定义在洛伦兹距离下的三角不等式是可以违反。这个性质使我们能够用不等式的符号来刻画双曲空间中点之间的成对关系。（双曲空间有几个重要的计算模型： the Poincare ball model, the hyper-boloid model, the Klein model）  
**Hyperboloid Model**  
&emsp;&emsp;定义 $\textbf{u},\textbf{v}\in \mathbb{R}^{n+1}$ 之间的洛伦兹内积如下：  
$$\langle \textbf{u},\textbf{v} \rangle_{\mathcal{L}} = -u_0v_0+\sum_{i=1}^nu_iv_i\tag{1} \label{eq1} $$
&emsp;&emsp;n维双曲面 $H^{n,\beta}\subseteq\mathbb{R}^{n+1}$由以下定义的点集组成：

$$H^{n,\beta}=\lbrace\textbf{x}\in \mathbb{R}^{n+1}:\Vert \textbf{x} \Vert_{\mathcal{L}}^2=-\beta,x_0>\beta \rbrace \tag{2} \label{eq2}$$

&emsp;&emsp;$\Vert \textbf{x} \Vert_{\mathcal{L}}^2=\langle \textbf{x},\textbf{x} \rangle_{\mathcal{L}}$ 表示向量X的洛伦兹范数。在此定义下，每个向量 $\textbf{x}\in H^{n,\beta}$ 的第0维度 $x_0$ 不能随意指定，应由以下公式定义：
$$x_0=\sqrt{\beta+\sum_{i=1}^n x_i}\tag{3} \label{eq3}$$
&emsp;&emsp;两点之间的相关测地距离为:  
$$d_l( \textbf{u},\textbf{v})=arccosh(-\langle\textbf{u},\textbf{v}\rangle_{\mathcal{L}}) \tag{4}\label{eq4}$$  
&emsp;&emsp;请注意，双曲面模型 $H^{n,\beta}$ 的原点向量为 $\boldsymbol{0}=(\beta,0,...,0)$ 并且 $\boldsymbol{0}$ 与任意向量 $\textbf{x}\in \mathbb{R}^{n+1}$ 的洛伦兹内积定义为 $\langle\textbf{0},\textbf{x}\rangle=-x_0<\beta$.这里原点向量的定义和原点向量与其他向量的内积应该是特殊定义，因为它们分别不满足等式 $\eqref{eq2}$ 和 $\eqref{eq1}$  
&emsp;&emsp;当 $\beta=1$时，该模型称为单位双曲面模型（unit Hyperboloid Model）。这将贯穿整个论文，并且将 $H^{n,1}$ 简记为 $H^{n}$。  

**Lorentz Distance**  
&emsp;&emsp;$\textbf{u}$ , $\textbf{v} \in H^n$ 之间的平方洛伦兹距离（简称洛伦兹距离）定义如下：  
$$d_{\mathcal{L}}^2(\textbf{u},\textbf{v})=\Vert \textbf{u}-\textbf{v}\Vert_{\mathcal{L}}^2=-2-2\langle\textbf{u},\textbf{v}\rangle_{\mathcal{L}}\tag{5}\label{eq5}$$  
&emsp;&emsp;它几乎满足欧几里得几何的所有公理，但是不满足三角不等式。三角不等式是正定黎曼度量的最关键几何性质之一，它表明对于任意三个点 $\textbf{x}$,$\textbf{y}$,$\textbf{z}$，任意两对点之间的距离$d(.,.)$的和大于或等于剩下一个点对之间的距离。
$$d(\textbf{x},\textbf{y})\leq d(\textbf{x},\textbf{z})+d(\textbf{z},\textbf{y})\tag{6}\label{eq6}$$
&emsp;&emsp;在双曲空间中，公式$\eqref{eq4}$定义的测地距离满足此不等式，但是在洛伦兹距离$\eqref{eq5}$下,可能不满足此不等式，因为黎曼度量是负的。考虑原点和两个点$\textbf{u}$,$\textbf{v}$组成的三角形，如图1所示。当两个点在$x_1$轴不同边相距很远时，违反了三角不等式。如果两个点在同$x_1$轴一边，三角不等式成立。  

![图一](triangle.png)  
<center>图1 (a)违反了三角不等式,（b）满足三角不等式 $\label{pic1}$</center>

### **Learning Triangle Inequalities**

&emsp;&emsp;Hsieh et al. (2017)指出学习嵌入空间中的距离而不是内积有利于学习细粒度的嵌入空间，该空间不仅可以捕获item-user交互的表示，还可以捕获item-item和user-user距离的表示。本质上，所谓的度量学习方案(metric learning scheme)受到三角不等式的约束。  
&emsp;&emsp;与协同度量学习方案相反，作者认为两点之间的特征交互可以通过洛伦兹距离的三角不等式的符号来学习，而不是使用距离本身。形式上，score function写为：
$$
\mathcal{T}(\textbf{x},\textbf{y})={d_{\mathcal{L}}^2(\textbf{x},\textbf{y})-d_{\mathcal{L}}^2(\textbf{x},\textbf{0})-d_{\mathcal{L}}^2(\textbf{0},\textbf{y}) \over \langle\textbf{0},\textbf{x}\rangle_{\mathcal{L}}\langle\textbf{0},\textbf{y}\rangle_{\mathcal{L}}}\tag{7}\label{eq7}
$$  
分子为不等式两边的差，分母是为了约束score function.  
&emsp;&emsp;该score function 对于所有的维度，取值范围均为[-0.5,2].因此，与协同度量学习方案相比，分数函数不受维度的影响。如图2（a），为了说明函数的布局，绘制了等式（7）在2D中两点的取值。  
![图2](2D.jpg)  
<center>图2 （a）triangle learning方案（例如，等式 $\eqref{eq7}$ ）和（b）在二维双曲面模型中使用测地距离的度量学习方案中分数函数的二维图</center>

因为二维双曲模型中的点只有一个自由参数（参考$\eqref{eq3}$），因此图中的x轴和y轴分别表示两个点的自由参数。同时，作者使用等式$\eqref{eq4}$中定义的测地距离绘制了协同度量学习方案的取值图2（b）。通过比较，可以观察到作者提出的方法的取值是平滑和有界的，但是协作度量学习方案的得分函数取值是无界的。有界性是有用的，因为嵌入向量可以自由地远离原点，而分数函数仍可以平滑增长。  

## Lorentzian Factorization Machine  

### **Overview**

![图3](pic3.jpg)  
<center>图3 LorentzFM结构图</center>

&emsp;&emsp;如图三所示，稀疏特征向量$\mathcal{V}_x$作为模型输入。经过Lorentz embedding layer，将所有特征投影到同一个双曲空间。接下来，将所有字段的嵌入引入一个新的triangle pooling层，该层作为所有特征对的聚合函数，从整体上度量三角形不等式的 soft “validness” 。与最近建立在欧几里德嵌入基础上的最先进的神经结构不同，LorentzFM不需要任何额外的参数。特别是，对于给定的稀疏输入$V_x$，池化层的输出是模型输出分数  
$$\hat{S}_{LFM}(\mathcal{V}_x)=\sum_{i,j=1,i\neq j}^d \mathcal{T} (\textbf{v}_i,\textbf{v}_j)x_ix_j \tag{8}\label{eq8} $$  
其中$\textbf{v}_i,\textbf{v}_j \in H^n$是每个输入特征字段的嵌入向量。$\mathcal{T}(.,.)$是特征交互函数。虽然在公式（8）中形式上缺少线性项，但它实际上在池函数$\mathcal{T}(.,.)$中重新出现，如下文所示。（为什么作者强调要有线性项）

### **Lorentz Embedding Layer**  

&emsp;&emsp;嵌入层是一个lookup操作，用于将稀疏特征投影到洛伦兹空间中的低维密集向量。即$\textbf{v}_k\in H^n$是第k个特征的嵌入向量，而第0个分量由等式$\eqref{eq3}$中的约束给出。  
&emsp;&emsp;在某些情况下，分类特征可以是多值的。例如，电影《泰坦尼克号》的类型可以是“Drama”或“Romance”。因此为这些分类特征使用多个字段，并用“unknown”填充它们，以确保每个样本在特征维度上对齐。

### **Triangle Pooling Layer**  

&emsp;&emsp;Pooling 层是一个聚合函数，用于将一组嵌入向量转换为一个向量：
$$
\mathcal{T}(\textbf{u},\textbf{v})={d_{\mathcal{L}}^2(\textbf{u},\textbf{v})-d_{\mathcal{L}}^2(\textbf{u},\textbf{0})-d_{\mathcal{L}}^2(\textbf{0},\textbf{v}) \over 2\langle\textbf{0},\textbf{u}\rangle_{\mathcal{L}}\langle\textbf{0},\textbf{v}\rangle_{\mathcal{L}}}
$$
$$
={1-\langle\textbf{u},\textbf{v}\rangle_{\mathcal{L}}-u_0-v_0 \over u_0v_0}
$$
$$
={1-\langle\textbf{u},\textbf{v}\rangle_{\mathcal{L}} \over 2u_0v_0}-\left({1\over u_0}+{1\over v_0}\right)\tag{9}\label{eq9}
$$

由于标准化分母，出现了线性项，如公式最后一行$\left({1\over u_0}+{1\over v_0}\right)$所示。  

### **Objective and Learning**  

&emsp;&emsp;推荐系统和CTR预测的目标函数是二元交叉熵（BCE）：
$$
\arg\,\min_{\theta}\sum_i-y_ilog(p_i)-(1-y_i)log(1-p_i)\tag{10}\label{eq10}
$$
i表示第i个样本，其中$p_i=\sigma\left(\hat{S}_{LFM}\left(\mathcal{V}_x^i\right)\right)$,为第i输入样本$\mathcal{V}_x^i$的（点击）可能性。$y_i$是真实标签。作者指出，尽管贝叶斯个性化排名（Bayesian Personalized Ranking, BPR）损失（Rendle et al.2009）在普遍的推荐系统中被证明是有用的，但作者不使用它，因为BCE损失是符号敏感的，这是期望的属性，而BPR损失不是。  
&emsp;&emsp;模型参数是通过使用黎曼随机梯度下降法（RSGD）（Bonnabel 2013）学习的。如Nickel和Kiela（2018）提到的，参数更新方式如下所示：
$$
\theta_{t+1}=exp_{\theta_t}(-\eta\ grad\ f(\theta_t))\tag{11}\label{eq11}
$$
其中，$grad\ f(\theta_t)$是黎曼流形中定义的梯度，$\eta$是学习率。黎曼梯度是通过将欧几里德空间中的梯度乘以洛伦兹度量，然后在当前参数集跨越的切线空间上执行正交投影来获得的。最后，通过以下指数映射给出参数更新:
$$
exp_{\theta_t}(\textbf{x})=cosh(\Vert\textbf{v}\Vert_{\mathcal{L}})\textbf{x}+sinh(\Vert\textbf{v}\Vert_{\mathcal{L}}){\textbf{v}\over \Vert\textbf{v}\Vert_{\mathcal{L}}}\tag{12}\label{eq12}
$$
它将切线空间中的切线向量v映射到洛伦兹流形上。详细信息见Nickel and Kiela（2018） Learning continuous hierarchies in the lorentz model of hyperbolic geometry。(待学习)  
&emsp;&emsp;由于score function等式$\eqref{eq7}$是有界且与维度无关的，因此无需在嵌入向量上应用L2正则化项，因为它会对双曲面模型的原点创造一个抵抗梯度。

## 实验

![表1](table1.jpg)
<center>表1 数据预处理后的数据集统计。在Avazu数据集中，itemID和userID没有明确的指示符，因此将相应的行留空。</center>

Steam, MovieLens and KKBox数据集用于推荐任务，Avazu用于CTR任务。

![表2](table2.jpg)
<center>表2 每个模型在测试集上的最佳性能。</center>

从表2可发现， 除了MovieLens数据集，LorentzFM的效果比其他模型都好。原因是MovieLens的稀疏特征的数量是其他数据集的5到100倍，说明当数据非常稀疏时，LorentzFM功能尤其强大。  

![表3](table3.jpg)
<center>表3 在最佳性能下，每个模型的训练参数数量与训练时间比较表。</center>

![图5](figure5.jpg)
<center>图5 Visualization of the heatmap of score functions from LorentzFM for (a) a positive sample and (b) a negative
sample, and from FM for the same (c) positive sample and
(d) negative sample</center>

&emsp;&emsp;作者选了一个典型的用户(user ID “76561198071045315”)。该用户几乎所有正评分都是Steam中免费的游戏。针对该用户，作者画了图5。从图5(a)中可以发现，对于游戏是免费的正样本，由LorentzFM学习到的“UserID” 和 “Price”的特征交互分数达到了最高0.4。说明用户非常倾向于免费的物品。而从图5（c）并不能得到该信息。  
&emsp;&emsp;对于游戏价格为\$14.99的负样本，由LorentzFM学习到的“UserID” 和 “Price”的特征交互分数几乎是最低的负数。这意味着负样本主要是由于其价格而被识别的。而FM中“UserID” 和 “Price” 的内积结果并不占主要地位。
