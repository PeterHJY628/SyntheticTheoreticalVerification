可以。若目标是复现这段 baseline theorem，我建议把“procedurally generated populations of \(10^6\) samples”实现成：每次实验程序生成一个 \(10^6\) 的 source population 和一个 \(10^6\) 的 shifted target population，用 Monte Carlo 近似两个真实分布。这样不会把小样本误差混进 theorem validation。

下面这个实现同时满足：

$$
p_S(x)\neq p_T(x),\qquad
p_S(y\mid x)=p_T(y\mid x),
$$

最优 Brier predictor 恰好属于 linear-softmax hypothesis class，

$$
h^*(x)=\operatorname{softmax}(W^*x),
$$

并且 source/target 均值严格接近 0，所以主要考察

$$
\Delta M
=
\mathbb E_T[XX^\top]-\mathbb E_S[XX^\top].
$$

```python
import math
import torch
import torch.nn.functional as F
from dataclasses import dataclass


# ============================================================
# Configuration
# ============================================================

@dataclass
class PopulationConfig:
    n_population: int = 1_000_000

    # severe covariate shift
    rotation_deg: float = 60.0
    scale_x: float = 4.0
    scale_y: float = 0.25

    # true conditional p(y|x)
    beta: float = 4.0

    # If your theorem requires ||x||_2 = 1
    l2_normalize: bool = True

    seed: int = 42


cfg = PopulationConfig()
```

先定义一个中心对称的 2D GMM。中心对称很有用，因为它可以消除 mean shift 这个混杂因素。

```python
# ============================================================
# Symmetric 2D GMM
# ============================================================

GMM_MEANS = torch.tensor([
    [ 2.4,  0.6],
    [-2.4, -0.6],
    [ 0.7,  1.6],
    [-0.7, -1.6],
], dtype=torch.float32)

GMM_WEIGHTS = torch.tensor([
    0.30,
    0.30,
    0.20,
    0.20,
], dtype=torch.float32)

GMM_COVS = torch.tensor([
    [[0.35,  0.08],
     [0.08,  0.15]],

    [[0.35,  0.08],
     [0.08,  0.15]],

    [[0.18, -0.03],
     [-0.03, 0.28]],

    [[0.18, -0.03],
     [-0.03, 0.28]],
], dtype=torch.float32)
```

采样时可以进一步用 antithetic sampling，即每生成一个 \(x\)，同时放入 \(-x\)。因为 GMM 本身是中心对称的，这不会改变目标分布，但会让有限的 \(10^6\) population 几乎严格满足

$$
\widehat{\mathbb E}[X]=0.
$$

```python
def sample_gmm(
    n: int,
    seed: int,
) -> torch.Tensor:
    """
    从中心对称 GMM 生成 n 个样本。

    使用 antithetic pairs:
        x, -x
    从而使 empirical mean 几乎严格为 0。
    """
    assert n % 2 == 0

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    half = n // 2

    chol = torch.linalg.cholesky(GMM_COVS)

    # Gaussian component
    z = torch.multinomial(
        GMM_WEIGHTS,
        half,
        replacement=True,
        generator=g,
    )

    eps = torch.randn(
        half,
        2,
        generator=g,
    )

    # L_z @ epsilon
    noise = torch.bmm(
        chol[z],
        eps.unsqueeze(-1),
    ).squeeze(-1)

    x_half = GMM_MEANS[z] + noise

    # exact central symmetry
    X = torch.cat([
        x_half,
        -x_half
    ], dim=0)

    return X
```

然后定义 severe rotation + anisotropic scaling：

```python
# ============================================================
# Covariate transformation
# ============================================================

def transformation_matrix(
    rotation_deg: float,
    scale_x: float,
    scale_y: float,
) -> torch.Tensor:

    theta = math.radians(rotation_deg)

    R = torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)],
    ], dtype=torch.float32)

    S = torch.tensor([
        [scale_x, 0.0],
        [0.0, scale_y],
    ], dtype=torch.float32)

    return R @ S
```

现在生成两个 \(10^6\) population。

```python
# ============================================================
# Generate source and target populations
# ============================================================

N = cfg.n_population

# Source population
X_source_raw = sample_gmm(
    N,
    seed=cfg.seed,
)

# Independent target base population
X_target_base = sample_gmm(
    N,
    seed=cfg.seed + 1,
)

A = transformation_matrix(
    cfg.rotation_deg,
    cfg.scale_x,
    cfg.scale_y,
)

# severe rotation + scaling
X_target_raw = X_target_base @ A.T


# ------------------------------------------------------------
# Optional l2 normalization
# ------------------------------------------------------------

if cfg.l2_normalize:
    X_source = F.normalize(
        X_source_raw,
        p=2,
        dim=1,
    )

    X_target = F.normalize(
        X_target_raw,
        p=2,
        dim=1,
    )
else:
    X_source = X_source_raw
    X_target = X_target_raw
```

这里有一个需要明确的细节。

如果设置

```python
l2_normalize = False
```

那么严格有

$$
M_T
\approx
A M_S A^\top.
$$

如果设置

```python
l2_normalize = True
```

则 target 实际变成

$$
x_T
=
\frac{Ax}{\|Ax\|_2},
$$

所以不再严格满足

$$
M_T=A M_S A^\top.
$$

但它符合你截图中的

$$
\|x\|_2=1
$$

假设。并且 anisotropic scaling 仍然会显著改变样本方向分布和

$$
\mathbb E[XX^\top].
$$

因此两种实验可以都做：

```text
GMM-covariance sanity check:   l2_normalize=False
theorem assumption experiment: l2_normalize=True
```

接下来是最关键的一点：不要根据 GMM component 决定类别。

我们单独规定一个完全固定的 linear-softmax conditional：

$$
p(y\mid x)
=
\operatorname{softmax}(W^*x).
$$

这样 source 和 target 使用同一个 \(p(y\mid x)\)。

```python
# ============================================================
# Exact linear-softmax ground truth
# ============================================================

w = torch.tensor(
    [1.0, -0.65],
    dtype=torch.float32,
)

w = w / torch.linalg.vector_norm(w)

beta = cfg.beta

# rows correspond to class 0 and class 1
#
# logit_1 - logit_0
# = beta * w^T x
#
W_star = torch.stack([
    -0.5 * beta * w,
     0.5 * beta * w,
])


def true_probabilities(
    X: torch.Tensor
) -> torch.Tensor:

    logits = X @ W_star.T

    return torch.softmax(
        logits,
        dim=1,
    )
```

因此 Bayes predictor 就是

$$
\boxed{
h^*(x)
=
\operatorname{softmax}(W^*x)
}
$$

它精确属于你的 hypothesis class。

所以在 Brier loss 下：

$$
\boxed{
\text{capacity slack}=0
}
$$

不是“近似为 0”，而是 population level 上理论上就是 0。

decision boundary 是：

$$
p(y=1\mid x)=p(y=0\mid x)
$$

等价于

$$
w^\top x=0.
$$

因此始终是一条严格的直线。

计算两个 population 的预测概率：

```python
P_source = true_probabilities(X_source)
P_target = true_probabilities(X_target)
```

注意这一步特别重要：

```python
P_source = true_probabilities(X_source)
P_target = true_probabilities(X_target)
```

两边使用的是同一个函数。

所以：

$$
\boxed{
p_S(y\mid x)=p_T(y\mid x)
}
$$

这才是 pure covariate shift。

如果 active learning 实验需要真正的 oracle label，可以这样生成：

```python
def sample_labels(
    probabilities: torch.Tensor,
    seed: int,
) -> torch.Tensor:

    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    u = torch.rand(
        probabilities.shape[0],
        generator=g,
    )

    y = (
        u < probabilities[:, 1]
    ).long()

    return y


y_source = sample_labels(
    P_source,
    cfg.seed + 10,
)

y_target_hidden = sample_labels(
    P_target,
    cfg.seed + 11,
)
```

`y_target_hidden` 只属于 simulator/oracle，active-learning algorithm 不应该看到。

---

然后直接验证所谓的 uncentered covariance shift。

```python
# ============================================================
# Uncentered second moment
# ============================================================

def uncentered_second_moment(
    X: torch.Tensor
) -> torch.Tensor:

    return X.T @ X / X.shape[0]


M_source = uncentered_second_moment(
    X_source
)

M_target = uncentered_second_moment(
    X_target
)

delta_M = M_target - M_source

moment_shift_fro = torch.linalg.matrix_norm(
    delta_M,
    ord="fro",
)


print("source mean:")
print(X_source.mean(dim=0))

print("\ntarget mean:")
print(X_target.mean(dim=0))

print("\nM_source:")
print(M_source)

print("\nM_target:")
print(M_target)

print("\nDelta M:")
print(delta_M)

print(
    "\n||Delta M||_F:",
    moment_shift_fro.item()
)
```

你希望看到的是：

$$
\widehat{\mathbb E}_S[X]\approx(0,0),
$$

$$
\widehat{\mathbb E}_T[X]\approx(0,0),
$$

但：

$$
\boxed{
\widehat M_S
\neq
\widehat M_T
}
$$

而且 severe shift 下：

$$
\boxed{
\|\widehat M_T-\widehat M_S\|_F
}
$$

应该明显大于 0。

这就是 “isolate uncentered covariance shift” 的第一个实验检查。

---

对于 Brier risk，甚至不需要 Monte Carlo 再抽 label。

如果真实 conditional probability 是 \(p(x)\)，预测也是 \(p(x)\)，则：

$$
\mathbb E_{Y\mid X=x}
\left[
\|p(x)-Y\|_2^2
\right]
=
1-\|p(x)\|_2^2.
$$

所以可以直接精确计算 \(10^6\) population 上的 expected Brier risk：

```python
# ============================================================
# Exact conditional Brier risk
# ============================================================

def bayes_brier_risk(
    probabilities: torch.Tensor,
) -> torch.Tensor:

    pointwise_risk = (
        1.0
        - probabilities.square().sum(dim=1)
    )

    return pointwise_risk.mean()


risk_source = bayes_brier_risk(
    P_source
)

risk_target = bayes_brier_risk(
    P_target
)

risk_gap = risk_target - risk_source


print(
    "R_source =",
    risk_source.item()
)

print(
    "R_target =",
    risk_target.item()
)

print(
    "R_target - R_source =",
    risk_gap.item()
)
```

这一步尤其适合你的 theorem validation，因为它完全去掉了 label sampling noise。

你现在有：

$$
\boxed{
\Delta R
=
R_T(h^*)-R_S(h^*)
}
$$

以及：

$$
\boxed{
\Delta M
=
M_T-M_S.
}
$$

接下来才让 critic 去估计 IPM。

---

你的 critic 代码可以直接接进去。

对于 theorem sanity check，我建议不要只用几十个 labeled points。因为那样你测试的同时还有 sampling error。

直接让 source population 和 target population 都是 \(10^6\)：

```python
# ============================================================
# Build critic population
# ============================================================

features = torch.cat([
    X_source,
    X_target,
], dim=0)

probabilities = torch.cat([
    P_source,
    P_target,
], dim=0)


labeled_idx = torch.arange(
    0,
    N,
    dtype=torch.long,
)

unlabeled_idx = torch.arange(
    N,
    2 * N,
    dtype=torch.long,
)
```

然后：

```python
critic = SpectralNormLayerIPM()

result = train_critic(
    critic=critic,
    features=features,
    probabilities=probabilities,
    labeled_idx=labeled_idx,
    unlabeled_idx=unlabeled_idx,
    config=config,
)
```

最后不要直接拿某个随机 validation minibatch 当最终 IPM。

训练结束以后重新对整个 \(10^6+10^6\) population 计算：

```python
# ============================================================
# Population IPM estimate
# ============================================================

critic.eval()

with torch.inference_mode():

    source_score = critic(
        X_source,
        P_source,
    ).mean()

    target_score = critic(
        X_target,
        P_target,
    ).mean()

    estimated_ipm = (
        target_score
        - source_score
    )


print(
    "critic IPM =",
    estimated_ipm.item()
)
```

如果担心一次 forward \(10^6\) 个点，可以 batch：

```python
def critic_mean(
    critic,
    X,
    P,
    batch_size=65536,
):
    total = 0.0
    n = 0

    critic.eval()

    with torch.inference_mode():

        for start in range(
            0,
            len(X),
            batch_size,
        ):
            end = min(
                start + batch_size,
                len(X),
            )

            score = critic(
                X[start:end],
                P[start:end],
            )

            total += score.sum().item()
            n += score.numel()

    return total / n


source_critic_mean = critic_mean(
    critic,
    X_source,
    P_source,
)

target_critic_mean = critic_mean(
    critic,
    X_target,
    P_target,
)

estimated_ipm = (
    target_critic_mean
    - source_critic_mean
)
```

最终一次实验至少记录这几个量：

```python
metrics = {
    "rotation_deg":
        cfg.rotation_deg,

    "scale_x":
        cfg.scale_x,

    "scale_y":
        cfg.scale_y,

    "source_mean_norm":
        X_source.mean(0).norm().item(),

    "target_mean_norm":
        X_target.mean(0).norm().item(),

    "moment_shift_fro":
        moment_shift_fro.item(),

    "source_brier_risk":
        risk_source.item(),

    "target_brier_risk":
        risk_target.item(),

    "risk_gap":
        risk_gap.item(),

    "critic_ipm":
        float(estimated_ipm),
}
```

那么你真正要验证的链路就是：

$$
\boxed{
\text{rotation/scaling severity}
\rightarrow
\|\Delta M\|_F
\rightarrow
\operatorname{IPM}
\rightarrow
\Delta R
}
$$

尤其建议 sweep：

```python
rotations = [
    0, 15, 30, 45, 60, 75
]

scales = [
    (1.0, 1.0),
    (1.5, 1 / 1.5),
    (2.0, 1 / 2.0),
    (3.0, 1 / 3.0),
    (4.0, 1 / 4.0),
]
```

其中

```python
rotation = 0
scale = (1, 1)
```

应该作为 control：

$$
\Delta M\approx0,
\qquad
\operatorname{IPM}\approx0.
$$

随着 anisotropic scaling / rotation 增强，应观察：

$$
\|\Delta M\|_F
$$

系统性变化，并检查 critic IPM 是否同步追踪。

还有一点与你当前代码直接有关：你现在

```python
input_dim = EMBEDDING_DIM + NUM_CLASSES
hidden_dim = input_dim // 2
```

对于这里：

$$
d=2,\quad C=2
$$

意味着

```python
input_dim = 4
hidden_dim = 2
```

也就是 critic 只有 **2 个 hidden units**。对于逼近你前面式 (7) 中的

$$
x^\top V^\top
J(\hat y)^2
Vx
$$

这种 quadratic interaction，容量非常小。做这个 theorem experiment 时，我会至少把 hidden width 固定到例如：

```python
hidden_dim = 32
```

或

```python
hidden_dim = 64
```

否则如果 IPM track 不好，你无法判断究竟是 theorem/critic 有问题，还是仅仅因为这个两神经元网络没能力逼近二次型。

另外，你的 `train_critic` 在 validation 时最好真正执行：

```python
critic.eval()
```

因为 `spectral_norm` 在 `train()` 状态的 forward 中会更新 power-iteration buffer。validation 后再：

```python
critic.train()
```

否则你现在所谓的 validation forward 本身仍可能改变 spectral-normalization 的内部状态。

如果是为了写 “utilizing procedurally generated populations of \(10^6\) samples” 这一版实验，以上这套更接近一个干净的 population-level theorem validation，而不是普通的小规模 toy dataset。
