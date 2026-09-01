可以。按照你前面的理论设定，我建议把主动学习实验定义得非常明确：

初始 labeled set 有 10 个 source-GMM 样本；unlabeled pool 是经过 rotation + anisotropic scaling 后的 \(10^6\) 个 target 样本；source/target 共用完全相同的 \(p(y\mid x)\)。每轮先训练 classifier，再训练 critic，记录指标，然后由 critic 从剩余 pool 中选 10 个点查询 oracle。20 轮后 labeled set 为

$$
10+20\times10=210.
$$

为了观察初始状态，建议记录 episode 0，因此最终得到 21 个 checkpoint：

$$
n_L=10,20,30,\ldots,210.
$$

最关键的是 True Risk Gap 的定义。因为你的 critic 优化的是

$$
\widehat{\operatorname{IPM}}
=
\mathbb E_U[f(x,\hat p)]
-
\mathbb E_L[f(x,\hat p)],
$$

所以对应的 True Risk Gap 也应该定义成

$$
\boxed{
\Delta R_t
=
R_{U_t}(h_t)-R_{L_t}(h_t)
}
$$

而不是随便拿 test accuracy 做差。

由于我们知道真实的

$$
p^*(y\mid x),
$$

Brier risk 可以不采样标签，直接精确计算：

$$
\ell_{\rm true}(h,x)
=
\mathbb E_{Y\mid x}
\|h(x)-Y\|_2^2
=
1+\|h(x)\|_2^2
-2h(x)^\top p^*(x).
$$

这样得到的才是真正适合 theorem validation 的 “True Risk”。

下面给你完整框架。

```python
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ============================================================
# 0. Config
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

N_POOL = 1_000_000
N_TEST = 1_000_000

INITIAL_LABELED = 10
QUERY_SIZE = 10
N_EPISODES = 20

FEATURE_DIM = 2
NUM_CLASSES = 2

SEED = 42


# ============================================================
# 1. Symmetric source GMM
# ============================================================

GMM_MEANS = torch.tensor([
    [ 2.4,  0.6],
    [-2.4, -0.6],
    [ 0.7,  1.6],
    [-0.7, -1.6],
], dtype=torch.float32)

GMM_WEIGHTS = torch.tensor(
    [0.30, 0.30, 0.20, 0.20],
    dtype=torch.float32,
)

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


def sample_symmetric_gmm(n, seed):
    """
    使用 antithetic sampling:
        x, -x

    因而 empirical mean 基本严格为 0。
    """
    assert n % 2 == 0

    g = torch.Generator()
    g.manual_seed(seed)

    half = n // 2

    z = torch.multinomial(
        GMM_WEIGHTS,
        half,
        replacement=True,
        generator=g,
    )

    L = torch.linalg.cholesky(GMM_COVS)

    eps = torch.randn(
        half,
        2,
        generator=g,
    )

    noise = torch.bmm(
        L[z],
        eps.unsqueeze(-1)
    ).squeeze(-1)

    x_half = GMM_MEANS[z] + noise

    return torch.cat(
        [x_half, -x_half],
        dim=0,
    )


# ============================================================
# 2. Severe rotation + scaling
# ============================================================

def transformation_matrix(
    angle_deg=60.0,
    scale_x=4.0,
    scale_y=0.25,
):
    theta = math.radians(angle_deg)

    R = torch.tensor([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta),  math.cos(theta)],
    ], dtype=torch.float32)

    S = torch.tensor([
        [scale_x, 0.0],
        [0.0, scale_y],
    ], dtype=torch.float32)

    return R @ S


A = transformation_matrix(
    angle_deg=60,
    scale_x=4.0,
    scale_y=0.25,
)


# ============================================================
# 3. Fixed true p(y | x)
# ============================================================

w_true = torch.tensor(
    [1.0, -0.65],
    dtype=torch.float32,
)

w_true /= torch.linalg.vector_norm(w_true)

BETA = 4.0

W_TRUE = torch.stack([
    -0.5 * BETA * w_true,
     0.5 * BETA * w_true,
])


def true_probability(X):
    """
    p*(y|x) = softmax(W* x)

    source 和 target 完全相同。
    """
    logits = X @ W_TRUE.T.to(X.device)

    return torch.softmax(
        logits,
        dim=1,
    )


def sample_oracle_labels(P, seed):
    """
    固定一次 oracle realization。
    """
    g = torch.Generator(device=P.device)
    g.manual_seed(seed)

    u = torch.rand(
        len(P),
        device=P.device,
        generator=g,
    )

    return (
        u < P[:, 1]
    ).long()


# ============================================================
# 4. Generate population
# ============================================================

# initial 10 source samples
X_initial = sample_symmetric_gmm(
    100,           # 先多生成一点
    SEED
)[:INITIAL_LABELED]


# 1e6 target AL pool
X_pool_base = sample_symmetric_gmm(
    N_POOL,
    SEED + 1,
)

X_pool = X_pool_base @ A.T


# independent 1e6 target evaluation population
X_test_base = sample_symmetric_gmm(
    N_TEST,
    SEED + 2,
)

X_test = X_test_base @ A.T


# ------------------------------------------------------------
# L2-normalized embedding
# ------------------------------------------------------------

X_initial = F.normalize(
    X_initial,
    p=2,
    dim=1,
)

X_pool = F.normalize(
    X_pool,
    p=2,
    dim=1,
)

X_test = F.normalize(
    X_test,
    p=2,
    dim=1,
)


# move to device
X_initial = X_initial.to(DEVICE)
X_pool = X_pool.to(DEVICE)
X_test = X_test.to(DEVICE)


# true conditional probability
P_initial_true = true_probability(X_initial)
P_pool_true = true_probability(X_pool)
P_test_true = true_probability(X_test)


# hidden oracle labels
y_initial = sample_oracle_labels(
    P_initial_true,
    SEED + 10,
)

y_pool_hidden = sample_oracle_labels(
    P_pool_true,
    SEED + 11,
)

y_test = sample_oracle_labels(
    P_test_true,
    SEED + 12,
)
```

这里 source 和 target 的区别仅仅在：

$$
p_S(x)\neq p_T(x),
$$

而标签机制严格相同：

$$
p_S(y\mid x)=p_T(y\mid x)
=
\operatorname{softmax}(W^*x).
$$

所以是我们需要的 pure covariate shift。

接下来训练 classifier。既然你的 theorem 用 Brier score，就不要用 cross entropy，直接用 Brier。

```python
# ============================================================
# 5. Linear classifier
# ============================================================

class LinearClassifier(nn.Module):

    def __init__(self):
        super().__init__()

        # 与理论 h(x)=softmax(Wx) 完全一致
        self.linear = nn.Linear(
            2,
            2,
            bias=False,
        )

    def forward(self, x):
        return torch.softmax(
            self.linear(x),
            dim=1,
        )


def fit_classifier(
    X_L,
    y_L,
    seed=0,
    max_steps=2000,
    lr=0.03,
):
    torch.manual_seed(seed)

    model = LinearClassifier().to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    Y = F.one_hot(
        y_L,
        num_classes=2
    ).float()

    previous = float("inf")
    stale = 0

    for step in range(max_steps):

        model.train()

        P = model(X_L)

        loss = (
            (P - Y).square()
            .sum(dim=1)
            .mean()
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        current = float(loss.detach())

        if abs(previous - current) < 1e-8:
            stale += 1
        else:
            stale = 0

        previous = current

        if stale >= 100:
            break

    model.eval()

    return model
```

然后定义真正的 population Brier risk。

```python
# ============================================================
# 6. Exact true Brier risk
# ============================================================

def pointwise_true_brier(
    prediction,
    true_probability,
):
    """
    E_{Y|x}[||prediction - onehot(Y)||^2]

    = 1 + ||q||^2 - 2 q^T p*
    """

    return (
        1.0
        + prediction.square().sum(dim=1)
        - 2.0 * (
            prediction
            * true_probability
        ).sum(dim=1)
    )
```

这一步非常重要。

不要计算：

```python
criterion(pred, sampled_y)
```

然后称其为 True Risk。

我们知道完整的 \(p^*(y\mid x)\)，因此可以直接积分掉 \(Y\) 的随机性。

你的 critic 建议稍微改一下 hidden width。二维问题里原来的

```python
hidden_dim = input_dim // 2
```

只有 2 个 hidden neurons，太小。

```python
# ============================================================
# 7. IPM critic
# ============================================================

class SpectralNormLayerIPM(nn.Module):

    def __init__(self, hidden_dim=64):
        super().__init__()

        input_dim = FEATURE_DIM + NUM_CLASSES

        self.net = nn.Sequential(

            spectral_norm(
                nn.Linear(
                    input_dim,
                    hidden_dim
                )
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True,
            ),

            spectral_norm(
                nn.Linear(
                    hidden_dim,
                    1
                )
            ),
        )

    def forward(
        self,
        features,
        probabilities,
    ):
        z = torch.cat(
            [features, probabilities],
            dim=1,
        )

        return self.net(z).squeeze(1)
```

下面给一个针对百万 pool 的 critic 训练函数。每一步只采 minibatch，不需要把全部百万样本送进去。

```python
def train_ipm_critic(
    critic,
    X_L,
    P_L,
    X_U,
    P_U,
    steps=2000,
    batch_size=1024,
    lr=1e-3,
):
    optimizer = torch.optim.AdamW(
        critic.parameters(),
        lr=lr,
        weight_decay=1e-4,
    )

    critic.train()

    best_obj = -float("inf")
    best_state = None

    patience = 20
    stale = 0

    for step in range(steps):

        l_idx = torch.randint(
            0,
            len(X_L),
            (batch_size,),
            device=DEVICE,
        )

        u_idx = torch.randint(
            0,
            len(X_U),
            (batch_size,),
            device=DEVICE,
        )

        s_L = critic(
            X_L[l_idx],
            P_L[l_idx],
        )

        s_U = critic(
            X_U[u_idx],
            P_U[u_idx],
        )

        objective = (
            s_U.mean()
            - s_L.mean()
        )

        loss = -objective

        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()
        optimizer.step()

        # validation estimate
        if (step + 1) % 50 == 0:

            critic.eval()

            with torch.no_grad():

                vl = torch.randint(
                    0,
                    len(X_L),
                    (min(4096, len(X_L)),),
                    device=DEVICE,
                )

                vu = torch.randint(
                    0,
                    len(X_U),
                    (4096,),
                    device=DEVICE,
                )

                obj = (
                    critic(
                        X_U[vu],
                        P_U[vu]
                    ).mean()
                    -
                    critic(
                        X_L[vl],
                        P_L[vl]
                    ).mean()
                ).item()

            critic.train()

            if obj > best_obj + 1e-5:

                best_obj = obj

                best_state = copy.deepcopy(
                    critic.state_dict()
                )

                stale = 0

            else:
                stale += 1

            if stale >= patience:
                break

    if best_state is not None:
        critic.load_state_dict(
            best_state
        )

    critic.eval()

    return critic
```

然后需要在完整的 remaining pool 上计算 IPM。百万样本不要一次 forward，分 batch。

```python
# ============================================================
# 8. Full population critic mean
# ============================================================

@torch.no_grad()
def critic_population_mean(
    critic,
    X,
    P,
    batch_size=65536,
):
    total = 0.0
    count = 0

    critic.eval()

    for start in range(
        0,
        len(X),
        batch_size
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
        count += len(score)

    return total / count
```

于是：

$$
\widehat{\mathrm{IPM}}_t
=
\frac1{|U_t|}
\sum_{x\in U_t}f_t(x)
-
\frac1{|L_t|}
\sum_{x\in L_t}f_t(x).
$$

```python
def evaluate_ipm(
    critic,
    X_L,
    P_L,
    X_U,
    P_U,
):
    mu_L = critic_population_mean(
        critic,
        X_L,
        P_L,
    )

    mu_U = critic_population_mean(
        critic,
        X_U,
        P_U,
    )

    return mu_U - mu_L
```

接下来是 acquisition。

由于你的 critic 被训练成：

$$
\max_f
\mathbb E_U[f]-\mathbb E_L[f],
$$

所以大的

$$
f(x)
$$

代表 critic 认为这个点更加属于当前 labeled set 没有覆盖的区域。

因此一种非常自然的 critic-based acquisition 是：

$$
\boxed{
x^*=\arg\max_{x\in U_t} f(x,\hat p(x)).
}
$$

每轮取 top 10。

```python
# ============================================================
# 9. Critic-based acquisition
# ============================================================

@torch.no_grad()
def select_topk_by_critic(
    critic,
    X_U,
    P_U,
    k=10,
    batch_size=65536,
):
    """
    返回 X_U 中的 local indices。
    """

    best_scores = None
    best_indices = None

    critic.eval()

    for start in range(
        0,
        len(X_U),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(X_U),
        )

        scores = critic(
            X_U[start:end],
            P_U[start:end],
        )

        kk = min(k, len(scores))

        values, idx = torch.topk(
            scores,
            k=kk,
        )

        idx = idx + start

        if best_scores is None:

            best_scores = values
            best_indices = idx

        else:

            all_scores = torch.cat([
                best_scores,
                values,
            ])

            all_indices = torch.cat([
                best_indices,
                idx,
            ])

            kk = min(
                k,
                len(all_scores),
            )

            best_scores, order = torch.topk(
                all_scores,
                k=kk,
            )

            best_indices = all_indices[
                order
            ]

    return best_indices
```

最后是整个 AL loop。

```python
# ============================================================
# 10. Active-learning experiment
# ============================================================

def run_active_learning():

    # --------------------------------------------------------
    # master population:
    #
    # [ initial source points | target pool ]
    # --------------------------------------------------------

    X_all = torch.cat([
        X_initial,
        X_pool,
    ])

    P_true_all = torch.cat([
        P_initial_true,
        P_pool_true,
    ])

    y_all_hidden = torch.cat([
        y_initial,
        y_pool_hidden,
    ])


    # Initial 10 source points
    labeled_idx = torch.arange(
        INITIAL_LABELED,
        device=DEVICE,
    )

    # target pool
    unlabeled_idx = torch.arange(
        INITIAL_LABELED,
        INITIAL_LABELED + N_POOL,
        device=DEVICE,
    )


    history = []


    # episode 0 = initial state
    # episode 1...20 = after 1...20 AL acquisitions
    for episode in range(
        N_EPISODES + 1
    ):

        print(
            f"\nEpisode {episode:02d} | "
            f"L={len(labeled_idx)} | "
            f"U={len(unlabeled_idx)}"
        )


        # ====================================================
        # A. Train classifier
        # ====================================================

        X_L = X_all[labeled_idx]
        y_L = y_all_hidden[labeled_idx]

        classifier = fit_classifier(
            X_L,
            y_L,
            seed=SEED + episode,
        )


        # ====================================================
        # B. Current predictions
        # ====================================================

        classifier.eval()

        with torch.no_grad():

            P_L_pred = classifier(
                X_all[labeled_idx]
            )

            P_U_pred = classifier(
                X_all[unlabeled_idx]
            )

            P_test_pred = classifier(
                X_test
            )


        # ====================================================
        # C. Accuracy
        # ====================================================

        y_test_pred = (
            P_test_pred.argmax(dim=1)
        )

        accuracy = (
            y_test_pred == y_test
        ).float().mean().item()


        # optional:
        # accuracy against Bayes decision boundary
        bayes_test_class = (
            P_test_true.argmax(dim=1)
        )

        boundary_accuracy = (
            y_test_pred
            == bayes_test_class
        ).float().mean().item()


        # ====================================================
        # D. TRUE risk gap
        # ====================================================

        true_risk_L = (
            pointwise_true_brier(
                P_L_pred,
                P_true_all[labeled_idx],
            )
            .mean()
            .item()
        )

        true_risk_U = (
            pointwise_true_brier(
                P_U_pred,
                P_true_all[unlabeled_idx],
            )
            .mean()
            .item()
        )

        true_risk_gap = (
            true_risk_U
            - true_risk_L
        )


        # ====================================================
        # E. Train IPM critic
        # ====================================================

        critic = SpectralNormLayerIPM(
            hidden_dim=64
        ).to(DEVICE)

        critic = train_ipm_critic(
            critic,

            X_all[labeled_idx],
            P_L_pred.detach(),

            X_all[unlabeled_idx],
            P_U_pred.detach(),
        )


        # ====================================================
        # F. Population IPM value
        # ====================================================

        ipm_value = evaluate_ipm(
            critic,

            X_all[labeled_idx],
            P_L_pred,

            X_all[unlabeled_idx],
            P_U_pred,
        )


        # ====================================================
        # G. Uncentered covariance diagnostic
        # ====================================================

        Xl = X_all[labeled_idx]
        Xu = X_all[unlabeled_idx]

        M_L = (
            Xl.T @ Xl
            / len(Xl)
        )

        M_U = (
            Xu.T @ Xu
            / len(Xu)
        )

        moment_gap = (
            torch.linalg.matrix_norm(
                M_U - M_L,
                ord="fro",
            ).item()
        )


        # ====================================================
        # H. Record
        # ====================================================

        row = {
            "episode":
                episode,

            "n_labeled":
                len(labeled_idx),

            "accuracy":
                accuracy,

            "boundary_accuracy":
                boundary_accuracy,

            "true_risk_L":
                true_risk_L,

            "true_risk_U":
                true_risk_U,

            "true_risk_gap":
                true_risk_gap,

            "abs_true_risk_gap":
                abs(true_risk_gap),

            "ipm":
                ipm_value,

            "moment_gap_fro":
                moment_gap,
        }

        history.append(row)

        print(row)


        # ====================================================
        # Stop after episode 20
        # ====================================================

        if episode == N_EPISODES:
            break


        # ====================================================
        # I. Active acquisition: query 10
        # ====================================================

        selected_local = (
            select_topk_by_critic(
                critic,
                X_all[unlabeled_idx],
                P_U_pred,
                k=QUERY_SIZE,
            )
        )


        # local U index -> global index
        selected_global = (
            unlabeled_idx[selected_local]
        )


        # add to labeled
        labeled_idx = torch.cat([
            labeled_idx,
            selected_global,
        ])


        # remove from U
        keep = torch.ones(
            len(unlabeled_idx),
            dtype=torch.bool,
            device=DEVICE,
        )

        keep[selected_local] = False

        unlabeled_idx = (
            unlabeled_idx[keep]
        )


    return history
```

运行：

```python
history = run_active_learning()
```

最后转成 DataFrame：

```python
import pandas as pd

df = pd.DataFrame(history)

print(
    df[
        [
            "episode",
            "n_labeled",
            "accuracy",
            "true_risk_gap",
            "ipm",
            "moment_gap_fro",
        ]
    ]
)

df.to_csv(
    "active_learning_gmm_ipm.csv",
    index=False,
)
```

最终你最应该画三条 trajectory。

第一张是 AL performance：

$$
n_L
\quad\text{vs}\quad
\mathrm{Accuracy}.
$$

第二张是 theorem validation：

$$
n_L
\quad\text{vs}\quad
\Delta R_t
$$

以及

$$
n_L
\quad\text{vs}\quad
\operatorname{IPM}_t.
$$

也就是直接把 `true_risk_gap` 和 `ipm` 画在同一张图。

如果 theorem 是

$$
\Delta R_t
\leq
\operatorname{IPM}_t
$$

而 capacity slack 为 0，你主要检查：

$$
\boxed{
\operatorname{IPM}_t-\Delta R_t\geq0
}
$$

以及这个 gap 是否足够小。

我建议额外记录：

```python
df["bound_slack"] = (
    df["ipm"]
    - df["true_risk_gap"]
)
```

第三张则是你那句 “track uncentered covariance shifts” 最关键的图：

$$
\boxed{
\|\widehat M_U-\widehat M_L\|_F
\quad\text{vs}\quad
\widehat{\operatorname{IPM}}
}
$$

如果随着 AL 不断选择 target samples：

$$
L_t
\leftarrow
L_t\cup Q_t,
$$

labeled distribution 应该逐渐接近 target distribution，因此理想情况下会看到：

$$
\|\widehat M_U-\widehat M_L\|_F
\downarrow,
$$

与此同时：

$$
\operatorname{IPM}_t
\downarrow,
$$

而：

$$
\mathrm{Accuracy}_t
\uparrow.
$$

这三者同时出现，才构成比较完整的证据链：

$$
\boxed{
\text{AL queries}
\rightarrow
\text{second-moment mismatch decreases}
\rightarrow
\text{IPM decreases}
\rightarrow
\text{true risk gap decreases}
\rightarrow
\text{target accuracy improves}
}
$$

还有一个实验设计上的重点：**不要只跑一个 seed。** 10 个初始标签的随机性非常大。论文实验至少建议跑 10–20 个 seeds，然后报告每个 episode 的 mean ± std；否则单条 AL trajectory 很容易偶然很好或偶然很差。

如果你的目标是严格验证 “Fisher-weighted IPM perfectly bounds the risk”，下一步最重要的不是继续改 GMM，而是确认你现在这个 `SpectralNormLayerIPM` 是否真的实现了 theorem 中定义的 **Fisher-weighted critic class**。单纯 spectral norm 本身只给出了 Euclidean Lipschitz constraint，这一点和 Fisher weighting 不能自动画等号。
