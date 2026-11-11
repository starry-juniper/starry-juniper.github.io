> **论文：** Improving ML Attacks on LWE with Data Repetition and Stepwise Regression  
> **作者：** Alberto Alfarano, Eshika Saxena, Emily Wenger, François Charton, Kristin Lauter  
> **会议：** ICML 2026  
> **研究方向：** Post-Quantum Cryptography（PQC）/ Machine Learning Cryptanalysis  
> **阅读类型：** 科研型粗读（Skimming）
>
> **关键词：** LWE、BKZ、Transformer、Data Repetition、Stepwise Regression、Scaling Law

## 1. 论文背景与研究问题

### 1.1 LWE 问题

Learning With Errors（LWE，容错学习问题）是格密码学中的重要数学困难问题，也是后量子密码学的重要理论基础之一。

LWE 的基本形式为：

$$
b=\langle\mathbf a,\mathbf s\rangle+e\pmod q
$$

其中：

- $\mathbf a$：已知随机向量；
- $\mathbf s$：未知秘密向量；
- $e$：较小的随机误差；
- $q$：模数；
- $b$：公开的计算结果。

攻击者获得大量 LWE 样本 $(\mathbf a_i,b_i)$，目标是恢复秘密向量 $\mathbf s$。

本文主要研究**稀疏二元和三元秘密向量**，即秘密中只有少量非零坐标。

例如：

$$
\mathbf s=(0,1,0,0,1,0,0,1)
$$

其汉明重量（Hamming Weight）为：

$$
h=3
$$

这里的汉明重量表示秘密向量中非零坐标的数量。

### 1.2 Cruel Region 与 Cool Region

已有研究发现，利用 BKZ 格约减算法预处理 LWE 样本后，不同坐标的约减效果存在明显差异。

因此可以将坐标分为两个区域：

| 区域 | 特点 | 恢复难度 |
|---|---|---|
| Cruel Region | 约减效果较差，坐标数值变化仍然较大 | 较困难 |
| Cool Region | 约减效果较好，坐标数值变化较小 | 相对容易 |

需要注意：Cruel 和 Cool 描述的是**经过约减后公开样本的坐标性质**，而 cruel bits、cool bits 指秘密向量中位于这些区域的坐标。

此前研究表明，当秘密向量中存在超过 3 个非零 cruel bits 时，机器学习攻击往往难以成功。

主要原因与模运算有关。

当非零秘密坐标增多时，内积

$$
\langle\mathbf a,\mathbf s\rangle
$$

可能多次超过模数 $q$，产生复杂的 modular wrap-around（模回绕）。

这使 Transformer 难以准确学习输入和输出之间的关系。

**本文的核心研究问题是：如何突破机器学习攻击难以恢复超过 3 个非零 cruel bits 的限制？**

---

## 2. Related Work：已有研究的发展

本文建立在 SALSA 系列机器学习密码分析方法之上。

### 2.1 SALSA（2022）

SALSA 是较早将 Transformer 用于 LWE 秘密恢复的工作。

基本思想是：

1. 收集共享同一秘密的 LWE 样本；
2. 训练 Transformer，根据 $\mathbf a$ 预测 $b$；
3. 比较相关输入下的模型预测，推断秘密坐标。

早期 SALSA 主要能够恢复汉明重量不超过 3 的秘密，且需要大量原始 LWE 样本。

### 2.2 PICANTE（2023）

PICANTE 引入了 BKZ 格约减预处理。

通过对 LWE 样本进行约减，降低部分公开向量坐标的数值变化幅度，使机器学习模型更容易学习相关的模运算。

这一改进扩大了可恢复秘密的范围。

### 2.3 VERDE（2023）

VERDE 进一步改进约减技术与秘密恢复方法，并将攻击扩展到三元秘密和其他小整数秘密。

### 2.4 FRESCA

FRESCA 改进了模型架构，使用 Encoder-only Transformer 和 Angular Embeddings（角度嵌入），以改善模型对模运算数据的处理。

### 2.5 Cool & Cruel（2024）

Cool & Cruel 从格约减的角度研究秘密恢复难度，发现不同坐标区域的约减效果明显不同。

由此形成了 Cruel Region 与 Cool Region 的划分，为解释机器学习攻击的瓶颈提供了重要依据。

### 2.6 本文所处的位置

可以将相关研究的技术发展概括为：

```text
SALSA
  │
  ├── PICANTE：引入 BKZ 预处理
  │
  ├── VERDE：改进约减和秘密恢复
  │
  ├── FRESCA：改进模型结构与嵌入
  │
  └── Cool & Cruel：分析不同坐标的恢复难度
          │
          ▼
       本文（2026）
          │
          ├── Larger Training Sets
          ├── Data Repetition
          └── Stepwise Regression
```

因此，本文并不是提出一套全新的 LWE 攻击框架，而是在已有机器学习攻击的基础上进一步改进训练策略与秘密恢复方法。

---

## 3. 核心创新方法

本文主要通过两条技术路线改善攻击能力：

- 利用更大的训练集和数据重复，提高 cruel bits 的恢复能力；
- 利用逐步回归，提高 cool bits 的恢复能力。

### 3.1 Larger Training Sets 与 Data Repetition

传统机器学习训练通常强调增加不同的训练样本。

但对于模运算学习，作者发现：**重复使用已有样本，也可能显著提高模型的学习效果。**

假设拥有 100 万条不同的 LWE 样本：

- 重复 1 次：模型处理 100 万个样本；
- 重复 4 次：模型总计处理 400 万个样本；
- 重复 10 次：模型总计处理 1000 万个样本。

虽然总训练量增加，但不同样本的数量没有变化。

作者通过实验发现，数据重复能够改善一些原本难以恢复的秘密的攻击效果。

例如，Section 4 的 Table 2 中，在  $n=64$、 $\log_2q=20$、 $h=5$ 的实验设置下，5000 万条不同样本对应的结果为：

| 重复次数 | 恢复结果 |
|---|---|
| 1× | 未成功 |
| 4× | 仅无噪声条件下成功 |
| 10× | 有噪声和无噪声条件下均成功 |

这说明重复训练能够帮助模型学习复杂的模运算关系。

不过，**数据重复不意味着训练成本一定降低**。它减少了对大量不同训练样本的依赖，但通常需要更多训练计算。

### 3.2 Stepwise Regression

当 cruel bits 已经被正确猜测后，可以从原始 LWE 结果中减去它们的贡献：

$$
b_{\mathrm{cool}}=
b-\mathbf a_{\mathrm{cruel}}\cdot
\mathbf s_{\mathrm{cruel}}
\pmod q
$$

剩下的任务就是恢复：

$$
\mathbf s_{\mathrm{cool}}
$$

此前方法使用 Linear Regression（线性回归），尝试同时估计所有 cool bits。

但这一方法存在两个问题：

1. 有限样本产生的协方差估计误差可能在矩阵求逆时被放大；
2. 普通线性回归没有正确处理模 $q$ 回绕，可能低估非零坐标的贡献。

因此作者提出 Stepwise Regression（逐步回归）。

其基本思想是：

**对于稀疏秘密，与其同时估计所有坐标，不如优先识别最有可能为零的坐标，并逐步排除。**

主要步骤如下：

1. 对剩余 cool bits 进行线性回归；
2. 归一化回归系数；
3. 找出绝对值最小的系数；
4. 将对应秘密坐标判定为零并移除；
5. 对剩余坐标重新回归；
6. 重复上述过程，直到满足秘密汉明重量约束。

由于秘密是稀疏的，大多数坐标通常为零，因此这一策略有可能更有效。

### 3.3 Dual Stepwise Regression

普通 Stepwise Regression 主要通过排除零坐标进行恢复。

然而，随着越来越多的零坐标被移除，剩余坐标中非零元素的比例会逐渐增加。

此时继续寻找零坐标可能更加困难。

因此作者提出 Dual Stepwise Regression（对偶逐步回归）。

对于二元秘密，可以考虑其补向量：

$$
\mathbf s'_{\mathrm{cool}}=
\mathbf 1-\mathbf s_{\mathrm{cool}}
$$

这样原来的 1 变成 0，原来的 0 变成 1。

通过在直接回归和对偶回归之间切换，可以改善秘密恢复效果。

**两种方法的核心区别是：**

- Stepwise Regression：优先排除原秘密中最可能为零的坐标；
- Dual Stepwise Regression：利用补向量，在剩余非零元素较多时转换恢复目标。

---

## 4. Experimental Settings 与实验结果

### 4.1 实验参数

论文主要考虑四组 LWE 参数：

| $n$ | $\log_2q$ | 特点 |
|---|---|---|
| 256 | 12 | Cruel Region 较大 |
| 512 | 28 | Cruel Region 较大 |
| 256 | 20 | BKZ 约减较充分 |
| 512 | 41 | BKZ 约减较充分 |

作者还使用了两种训练数据。

**BKZ-reduced Data**

通过实际 BKZ 预处理产生的约减数据。

**Synthetic Data**

根据实际约减样本测得的统计特征，人工生成具有类似分布的数据。

这样做的原因是 BKZ 计算成本很高，而生成大量合成数据的代价相对较低。

论文在测试设置下观察到，两种数据训练出的模型具有相近的秘密恢复表现。

### 4.2 攻击流程

整体攻击可以概括为：

```text
LWE 样本
   │
   ▼
BKZ 约减 / 合成数据生成
   │
   ▼
Transformer 训练
   │
   ▼
利用模型预测差异猜测 cruel bits
   │
   ▼
Stepwise / Dual Stepwise Regression
   │
   ▼
恢复 cool bits
   │
   ▼
验证完整秘密向量
```

需要注意，Transformer 并不是直接输出完整秘密向量，而是通过预测 LWE 结果，辅助后续的秘密恢复算法。

### 4.3 最大可恢复汉明重量

论文报告的部分二元秘密实验结果如下：

| 参数设置 | Prior ML | 本文 |
|---|---:|---:|
| $n=256,\log_2q=20$ | 33 | 70 |
| $n=512,\log_2q=41$ | 63 | 75 |

其中，最大可恢复汉明重量从 33 提高到 70 的结果尤其明显。

论文还报告了最多恢复 **8 个非零 cruel bits** 的实验结果。

但需要注意，这里报告的是特定实验预算下取得的最佳恢复能力，不能据此认为所有同样汉明重量的秘密都可以被恢复。

### 4.4 期望秘密恢复率

为了更合理地评价攻击性能，作者不仅研究最大可恢复汉明重量，还估计了给定汉明重量下的期望恢复率：

$$
E(h)=\sum_k p(h,k)r(h,k)
$$

其中：

- $p(h,k)$：汉明重量为 $h$ 的随机秘密具有 $k$ 个非零 cruel bits 的概率；
- $r(h,k)$：在对应条件下的秘密恢复率；
- $E(h)$：对不同 cruel bits 数量加权后的期望恢复率。

例如：

| 实验设置 | VERDE | 本文 |
|---|---:|---:|
| $n=256,\log_2q=20,h=33$ | 33% | 98% |
| $n=512,\log_2q=41,h=63$ | 15% | 91% |

这些结果说明，本文不仅提高了最大可恢复汉明重量，还改善了实验设置下对随机稀疏秘密的预期恢复能力。

这里的百分比是作者根据条件恢复率与秘密分布估计的结果，而不是单次训练的成功概率。

### 4.5 Stepwise Regression 的实验效果

论文 Table 5 对比了三种方法：

- Linear Regression；
- Stepwise Regression；
- Dual Stepwise Regression。

在 $n=256,\log_2q=20$、已知 cruel bits、使用 2000 万条恢复样本的设置下，当秘密具有 45 个非零 cool bits 时：

| 方法 | 成功恢复数量（20个秘密） |
|---|---:|
| Linear Regression | 0 |
| Stepwise Regression | 3 |
| Dual Stepwise Regression | 18 |

结果表明，在所测试的 LWE 数据分布下，Dual Stepwise Regression 能显著改善 cool bits 的恢复效果。

但该实验假设 cruel bits 已知，因此不能直接把表中的成功数量理解为完整攻击的成功率。

---

## 5. Scaling Laws 与研究局限

### 5.1 Scaling Laws

作者进一步研究模型规模、训练数据量、数据重复次数与攻击效果的关系。

实验发现，在所测试的模型规模范围内，单纯扩大 Transformer 参数量并没有明显改善攻击效果。

相比之下，数据量与重复次数对恢复能力的影响更值得关注。

作者提出经验关系：

$$
\ln A_R=C_R-\alpha_R\ln D
$$

其中：

- $A_R$：找到正确 cruel bits 所需的模型候选尝试次数；
- $D$：总训练样本处理量；
- $R$：数据重复次数；
- $C_R,\alpha_R$：由实验拟合得到的参数。

Table 13 中，重复次数与拟合指数的关系为：

| $R$ | $\alpha_R$ |
|---|---:|
| 1 | 0.70 |
| 2 | 1.31 |
| 5 | 1.45 |
| 15 | 1.58 |
| 50 | 1.95 |

这一结果提示：数据重复不仅减少了对不同样本的需求，还可能改变攻击性能随训练数据量增加而改善的速度。

但是，这条 Scaling Law 主要基于单个秘密向量的实验拟合，尚不能将其视为适用于所有 LWE 参数的普遍规律。

### 5.2 论文的局限

作者在 Discussion 中明确指出了以下问题。

**（1）攻击范围有限**

本文主要针对基本形式的 LWE，以及稀疏二元、三元秘密。

不能直接将结果推广到 Ring-LWE、Module-LWE 或实际标准化 PQC 方案。

**（2）计算资源消耗巨大**

大规模 BKZ 数据预处理十分昂贵。

例如，论文报告某组 4 亿条约减数据消耗约 4040 万 CPU 核时。

这会影响攻击的实际可行性。

**（3）缺少完整的理论解释**

目前尚未严格刻画 Data Repetition 与 Stepwise Regression 在什么条件下成功或失败。

**（4）Scaling Law 的泛化性有待验证**

现有拟合实验覆盖的秘密向量数量有限。

是否能够迁移到其他维度、模数、稀疏程度和秘密分布，仍然需要进一步实验。

另外，Section 6 的整体秘密恢复实验还假设攻击者知道秘密的汉明重量 $h$，这一前提也应在评价攻击能力时考虑。

---

## 6. 科研启发与粗读总结

### 6.1 潜在研究方向

通过粗读，我认为这篇论文至少提供了三个值得进一步探索的方向。

#### 方向一：Data Repetition 的泛化能力

本文发现数据重复能够改善秘密恢复效果，但其 Scaling Law 尚缺少广泛验证。

可以考虑改变：

- LWE 维度 $n$；
- 模数 $q$；
- 秘密汉明重量 $h$；
- 秘密分布；
- 训练数据量与重复次数；
- 随机秘密及模型初始化。

观察攻击成功率与经验缩放规律是否依然成立。

这一方向适合从实验验证入手。

#### 方向二：合成数据预训练

由于实际 BKZ 约减成本很高，可以研究先使用 Synthetic Data 进行预训练，再通过少量实际约减数据进行微调。

其核心研究问题是：

**能否在降低数据预处理成本的同时，保持甚至改善秘密恢复能力？**

不过，合成数据与真实数据的分布差异，以及这种差异对攻击性能的影响，需要进一步验证。

#### 方向三：Stepwise Regression 的理论分析

本文表明逐步回归在部分稀疏 LWE 实验中效果较好，但缺少完整理论保证。

可以研究：

- 何种条件下能够正确识别零坐标？
- 样本数量与恢复成功率之间存在什么关系？
- 噪声大小如何影响回归系数的可靠性？
- 能否建立逐步回归的成功概率界？

这一方向更加偏向数学理论分析。

### 6.2 本次粗读的主要收获

通过六个阶段的学习，我形成了以下认识：

**第一，机器学习密码分析并不意味着直接利用神经网络输出私钥。**

本文通过 Transformer 学习 LWE 样本中的模运算关系，再结合区分器和回归算法恢复秘密。

**第二，算法性能的提高不一定依赖更大的神经网络。**

训练数据的组织方式与后处理算法也可能显著影响攻击效果。

**第三，需要区分最佳实验结果与整体攻击成功率。**

某个方法能够恢复较高汉明重量的秘密，不代表它对所有相同汉明重量的秘密都有效。

**第四，经验实验与理论保证是不同层次的研究结果。**

Scaling Law 提供了有价值的实验观察，但目前还不能被当成已经证明的普遍规律。

**第五，研究局限本身也可能是科研选题的起点。**

例如，数据重复的泛化性、合成数据的有效性以及逐步回归的理论成功条件，都值得进一步调查。

### 6.3 是否值得精读？

我的判断是：**值得保留为 AI + PQC 交叉研究的候选论文，但暂时没有必要全文精读。**

如果后续决定开展相关研究，可以有针对性地精读：

| 章节 | 关注内容 |
|---|---|
| Section 4 | Data Repetition 与训练数据策略 |
| Section 5 | Stepwise Regression 算法 |
| Section 7 | Scaling Laws 与泛化问题 |

在选题尚未明确时，继续粗读其他相关论文，有助于建立更完整的研究背景。

## 7. 总结

本文通过扩大训练集、重复使用训练样本，以及引入 Stepwise Regression，显著改善了机器学习对稀疏 LWE 秘密的恢复能力。

其重要启发是：

> 对于机器学习密码分析，训练数据策略与秘密恢复算法可能和神经网络架构本身同样重要。

不过，本文的攻击结果主要建立在特定 LWE 参数、秘密分布和计算预算之上，其理论解释与实际密码方案适用性仍需要进一步研究。

从科研角度看，这篇论文不仅展示了一种有效的实验改进方法，也提出了值得继续探索的数据策略、数学理论与泛化性问题。

---

## 参考文献

[1] Alfarano, A., Saxena, E., Wenger, E., Charton, F., & Lauter, K. *Improving ML Attacks on LWE with Data Repetition and Stepwise Regression*. ICML, 2026.

[2] Wenger, E., et al. *SALSA: Attacking Lattice Cryptography with Transformers*. NeurIPS, 2022.

[3] Li, C. Y., et al. *SALSA PICANTE: A Machine Learning Attack on LWE with Binary Secrets*. ACM CCS, 2023.

[4] Li, C., et al. *SALSA VERDE: A Machine Learning Attack on Learning With Errors with Sparse Small Secrets*. NeurIPS, 2023.

[5] Nolte, N., et al. *The Cool and the Cruel: Separating Hard Parts of LWE Secrets*. AFRICACRYPT, 2024.

[6] Stevens, S., et al. *SALSA FRESCA: Angular Embeddings and Pre-training for ML Attacks on Learning With Errors*. IACR ePrint 2024/150.