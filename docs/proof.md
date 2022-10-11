---
abstract: |
  Convergence proof.
author:
- Charles Laroche, Andrés Almansa, Eva Coupeté, Matias Tassano
title: Provably Convergent Plug & Play Linearized ADMM, applied to
  Deblurring Spatially Varying Kernels
---

# Convergence of linearized-ADMM

In this section, we prove Theorem 1, without loss of generality we
suppose that $\lambda=1$ in our MAP estimator defined in ().

::: {#lemma:lagr_x .lemma}
**Lemma 1**. *Under assumption 1, the following inequality holds for the
x-update:
$$\mathcal{L}_\beta(x_k,z_k,w_k) - \mathcal{L}_\beta(x_{k+1},z_k,w_k) \geq \frac{L_x - \beta \|H\|^2}{2}\|x_k - x_{k+1}\|^2$$
with $\|H\|^2$ the largest singular value of $H^T H$.*
:::

::: {.proof}
*Proof.* Using the notation
of [\[equ:approx_lagr\]](#equ:approx_lagr){reference-type="ref"
reference="equ:approx_lagr"}, we define
$\overline{f}^k (x) = \Tilde{\mathcal{L}}^k_{\beta}(x,z_k,w_k)$ and by
definition of the $x$-update, we have: $$\begin{aligned}
        & \overline{f}^k(x_k) \geq \overline{f}^k(x_{k+1}) \\
        \Leftrightarrow & \langle x_k - x_{k+1} , H^T w_k + \beta H^T(Hx_k - z_k)\rangle \nonumber \\
         \label{equ:fk}
         & + f(x_k) - f(x_{k+1}) \geq   \frac{L_x}{2}\|x_{k+1}-x_k\|^2
    \end{aligned}$$ We also have that: $$\begin{aligned}
        & \mathcal{L}_\beta(x_k,z_k,w_k) - \mathcal{L}_\beta(x_{k+1},z_k,w_k)  \nonumber \\
        & =f(x_k) - f(x_{k+1}) + \langle w_k, H(x_k-x_{k+1}) \rangle \nonumber \\
        & \ \  + \frac{\beta}{2}\|Hx_k-z_k\|^2 - \frac{\beta}{2}\|Hx_{k+1} - z_k\|^2  \\
        &=  f(x_k) - f(x_{k+1}) - \frac{\beta}{2}\|H(x_{k+1} -x_k) \|^2 \nonumber \\
        & \ \ + \langle x_k - x_{k+1}, H^T w_k + \beta H^T(Hx_k - z_k) \rangle \\
        & \geq  \frac{L_x}{2}\|x_{k+1}-x_k\|^2 - \frac{\beta}{2}\|H(x_{k+1} -x_k) \|^2  \label{equ:lemma_3_1}  \\
        & \geq \frac{L_x-\beta \|H\|^2}{2}\|x_{k+1} -x_k\|^2
    \end{aligned}$$ where the inequality
[\[equ:lemma_3\_1\]](#equ:lemma_3_1){reference-type="eqref"
reference="equ:lemma_3_1"} is obtained using
[\[equ:fk\]](#equ:fk){reference-type="eqref" reference="equ:fk"}. ◻
:::

::: {#lemma:lagr_z .lemma}
**Lemma 2**.
*$\mathcal{L}_\beta(x_{k+1},z_k,w_k) - \mathcal{L}_\beta(x_{k+1},z_{k+1},w_k) \geq m\|z_k - z_{k+1}\|^2$*
:::

::: {.proof}
*Proof.* From Assumption 1, we have that $\mathcal{L}_\beta$ is strongly
convex in z with parameter m. The strong convexity of
$\mathcal{L}_\beta$ implies that: $$\begin{aligned}
        & \mathcal{L}_\beta(x,z_k,u) - \mathcal{L}_\beta(x,z_{k+1},u) \\
        & \ \ \ \ \geq \nabla_z \mathcal{L}_\beta(x,z_{k+1},u)(z_k - z_{k+1}) + m \|z_k - z_{k+1}\|^2 
    \end{aligned}$$ However, the z-update of Algorithm 1 is such that
$$\nabla_z \mathcal{L}_\beta(x_{k+1},z_{k+1},w_k) = 0$$ which leads to
the results. ◻
:::

::: {#lemma:w_grad_h .lemma}
**Lemma 3**. *Under Assumption 1, the following equality holds:
$$w_k = \nabla_z h(z_k)$$*
:::

::: {.proof}
*Proof.* From the definition of the Lagrangian:
$$\nabla_z \mathcal{L}_\beta(x,z,w) = \nabla h(z) - w - \beta (Hx-z)$$
Using the fact that
$$w_{k+1}= w_k + \beta (Hx_{k+1}-z_{k+1}) \quad \text{and} \quad \nabla_z \mathcal{L}_\beta(x_{k+1},z_{k+1},w_k) =0$$
We have: $$\begin{aligned}
        & 0 =  \nabla h(z_{k+1}) - w_k - \beta (Hx_{k+1}-z_{k+1}) \\
        \Leftrightarrow &  \nabla h(z_{k+1}) = w_{k+1}
    \end{aligned}$$ ◻
:::

::: {#lemma:lagr_u .lemma}
**Lemma 4**. *Under assumption 1, $$\begin{aligned}
   & \mathcal{L}_\beta(x_{k+1},z_{k+1},w_{k+1}) - \mathcal{L}_\beta(x_{k+1},z_{k+1},w_k)  \\
   & =  \frac{1}{\beta}\|w_{k+1} - w_k\|^2  \leq  C_1\|z_{k+1}-z_k\|^2\end{aligned}$$
with $C_1 = L_h^2/\beta$.*
:::

::: {.proof}
*Proof.* By definition of the augmented Lagrangian we have that:
$$\begin{aligned}
        \mathcal{L}_\beta(x_{k+1},z_{k+1},w_{k+1}) & - \mathcal{L}_\beta(x_{k+1},z_{k+1},w_k) \\
        & = \langle w_{k+1} - w_k, H x_{k+1}  - z_{k+1}\rangle \\
        & = \frac{1}{\beta}\|w_{k+1}-w_k\|^2 \\
        & = \frac{1}{\beta}\|\nabla_z h(z_{k+1}) - \nabla_z h(z_{k})\|^2 \ \ \text{ from Lemma~\ref{lemma:w_grad_h}} \\
        & \leq \frac{L_h^2}{\beta}\|z_{k+1}-z_k\|^2 \ \ \text{\hfill from \ Assumption~1}%Equation~\ref{equ:gradL} 
    \end{aligned}$$ ◻
:::

::: {#lemma:lipsch .lemma}
**Lemma 5**. *Let $g$ is $L_g$-Lipschitz differentiable then:
$$g(y_2) - g(y_1) \geq \nabla g(s)(y_2 - y_1) - \frac{L_g}{2}\|y_2-y_1\|^2$$
where s denotes $y_1$ or $y_2$*
:::

::: {.proof}
*Proof.* $$\begin{aligned}
& g(y_2)-g(y_1)
= \int_0^1 \nabla g(ty_2+(1-t) y_1) \cdot(y_2-y_1) \mathrm{d} t \\
=& \int_0^1 \nabla g(s) \cdot(y_2-y_1) \mathrm{d} t+\int_0^1(\nabla g(y_2+(1-t) y_1)-\nabla g(s))\cdot(y_2-y_1) \mathrm{d} t,\end{aligned}$$
where $\nabla g(\cdot)$ defines the gradient of $g(\cdot)$. If we take
$s=y_1$, then by inequality
$$\|\nabla g(t y_2+(1-t) y_1)-\nabla g(y_1)\| \leq L_g\|t(y_2-y_1)\|$$
we have $$\begin{aligned}
& \int_0^1 \nabla g(y_1) \cdot(y_2-y_1) \mathrm{d} t+\int_0^1(\nabla g(t y_2+(1-t) y_1)-\nabla g(y_1)) \cdot(y_2-y_1) \mathrm{d} t \\
\geq & \nabla g(y_1) \cdot(y_2-y_1)-\int_0^1 L_g t\|y_2-y_1\|^2 \mathrm{~d} t \\
=& \nabla g(y_1) \cdot(y_2-y_1)-\frac{L_g}{2}\|y_2-y_1\|^2 .\end{aligned}$$
Therefore, we get
$$g(y_2)-g(y_1) \geq \nabla g(y_1) \cdot(y_2-y_1)-\frac{L_g}{2}\|y_2-y_1\|^2 .$$
Similarly, if we take $s=y_2$, we can get
$$g(y_2)-g(y_1) \geq \nabla g(y_2) \cdot(y_2-y_1)-\frac{L_g}{2}\|y_2-y_1\|^2 .$$ ◻
:::

::: {#lemma:mk-converges .lemma}
**Lemma 6**. *Under Assumption 1, if we choose the hyper-parameters
$\beta$ and $L_x$ satisfying
([\[eq:convergence-condition1\]](#eq:convergence-condition1){reference-type="ref"
reference="eq:convergence-condition1"}) and
[\[eq:convergence-condition2\]](#eq:convergence-condition2){reference-type="eqref"
reference="eq:convergence-condition2"}, then the sequence $\{m_k\}$
defined by $$m_k = \mathcal{L}_\beta(x_k,z_{k},w_{k})$$ is convergent.*
:::

::: {.proof}
*Proof.* 1) **Monotonicity:** By using
Lemma [Lemma 1](#lemma:lagr_x){reference-type="ref"
reference="lemma:lagr_x"},
Lemma [Lemma 2](#lemma:lagr_z){reference-type="ref"
reference="lemma:lagr_z"} and
Lemma [Lemma 4](#lemma:lagr_u){reference-type="ref"
reference="lemma:lagr_u"} we have: $$\begin{aligned}
        & m_{k} - m_{k+1} = \mathcal{L}_\beta(x_{k}, z_{k}, w_{k}) - \mathcal{L}_\beta(x_{k+1}, z_{k+1}, w_{k+1}) \\
        & \ \ \ \ \geq \mathcal{L}_\beta(x_{k+1}, z_{k}, w_{k}) - \mathcal{L}_\beta(x_{k+1}, z_{k+1}, w_{k+1}) \\
        & \ \ \ \ \ \ \ +  \frac{L_x-\beta \|H\|^2}{2} \|x_k - x_{k+1}\|^2 \\
        & \ \ \ \ \geq \mathcal{L}_\beta(x_{k+1}, z_{k+1}, w_{k}) - \mathcal{L}_\beta(x_{k+1}, z_{k+1}, w_{k+1}) \\
        & \ \ \ \ \ \ \ +  \frac{L_x-\beta\|H\|^2}{2} \|x_k - x_{k+1}\|^2  + m \|z_{k}-z_{k+1}\|^2 \\
        & \label{equ:monot_mk_1}
        \ \ \ \ \geq  \frac{L_x-\beta\|H\|^2}{2} \|x_k - x_{k+1}\|^2 + (m + \frac{L_h^2}{\beta})\|z_{k}-z_{k+1}\|^2
    \end{aligned}$$ Since we chose $L_x$ such that: $$\begin{aligned}
        & L_x \geq  \beta\|H\|^2\\
        \Leftrightarrow \quad  & \frac{L_x-\beta\|H\|^2}{2} > 0
    \end{aligned}$$ we obtain the monotonocity of $\{m_k\}$. ) **Lower
bound**: $$\begin{aligned}
        m_k & =  h(z_k) + f(x_k) + \langle w_k, H x_k-z_k\rangle + \frac{\beta}{2}\|Hx_k-z_k\|^2
    \end{aligned}$$ Let $z_{k}' = H x_k$, from
Lemma [Lemma 3](#lemma:w_grad_h){reference-type="ref"
reference="lemma:w_grad_h"} we have: $$\begin{aligned}
        \langle w_k, H x_k-z_k\rangle 
        & = \langle w_k, z_{k}' - z_k\rangle \\
        & = \langle \nabla h(z_k) , z_{k}' - z_k\rangle
    \end{aligned}$$ so we can rewrite: $$\begin{aligned}
        m_k & =  h(z_k) + f(x_k) + \langle \nabla h(z_k) , z_k' - z_k\rangle + \frac{\beta}{2}\|z_k'-z_k\|^2 \\
    \end{aligned}$$ We chose $\beta$ such that $\beta \geq L_h$ so:
$$\begin{aligned}
        m_k & \geq  h(z_k) + f(x_k) - \langle \nabla h(z_k) , z_k - z_k'\rangle + \frac{L_h}{2}\|z_k - z_k'\|^2 \\
        & \geq h(z_k') + f(x_k) \ \ \text{from Lemma~\ref{lemma:lipsch}.}
    \end{aligned}$$ Following Assumption 1, $h(z_k') + f(x_k)$ is lower
bounded so ${m_k}$ is lower bounded. ${m_k}$ is monotonically decreasing
and lower bounded which ensure the convergence. ◻
:::

::: {#lemma_f1_f2 .lemma}
**Lemma 7**. *Suppose we have a differentiable function $f_1$, a
possibly non differentiable function $f_2$, and a point x. If there
exist $d2 \in \partial f_2(x)$, then we have:
$$d=d_2 - \nabla f_1(x) \in \partial(f_2(x) - f_1(x))$$*
:::

::: {.proof}
*Proof.* From the subgradient definition we have that:
$$f_2(y) \geq f_2(x) + \langle d_2, y-x \rangle + o(\|y-x\|)$$ From the
fact that $f_1$ is differentiable we have that:
$$-f_1(y) = -f_1(x) - \langle\nabla f_1(x), y-x\rangle + o(\|y-x\|)$$
Combining the two leads to:
$$f_2(y) -f_1(y) \geq f_2(x) - f_1(x) + \langle d_2 - \nabla f_1(x), y-x\rangle + o(\|y-x\|)$$ ◻
:::

*Proof of Theorem 1:* **Convergence of the residuals:** From
Lemma [Lemma 6](#lemma:mk-converges){reference-type="ref"
reference="lemma:mk-converges"} and its proof we have that:
$$m_{k+1} - m_k \geq a \|x_k - x_{k+1}\|^2 + m \|z_{k-1}-z_{k}\|^2 \geq 0$$
with $(m + \frac{L_h^2}{\beta}) > 0$,
$a = \frac{L_x-\beta \|H\|^2}{2} >0$ (according to Assumption 1) and
that $m_k$ converges. This implies that $\|x_k - x_{k+1}\|^2$ and
$\|y_k - y_{k+1}\|^2$ converge to 0 as k approaches infinity.
Lemma [Lemma 4](#lemma:lagr_u){reference-type="ref"
reference="lemma:lagr_u"} ensure the convergence of
$\|w_k - w_{k+1}\|^2$ to 0. The convergence of $m_k$ directly implies
the convergence of $\mathcal{L}_\beta(x_k,z_{k},w_{k})$. **Convergence
of the gradients:** For the convergence of
$\lim_{k\to\infty} \nabla_u \mathcal{L}_\beta(x_k,z_k,w_k)$, we have
that:
$$\lim_{k\to\infty} \nabla_u \mathcal{L}_\beta(x_k,z_k,w_k) =  \lim_{k\to\infty} Hx_k - z_k = \lim_{k\to\infty}\frac{1}{\beta}(w_{k+1}-w_k) =0.$$
On the other side, we have using
Lemma [Lemma 3](#lemma:w_grad_h){reference-type="ref"
reference="lemma:w_grad_h"} that: $$\begin{aligned}
     \nabla_z \mathcal{L}_\beta(x_k,z_k,w_k) & = \nabla h(z_k) - w_k - \beta (Hx_k - z_k) \\
     & = w_k - w_k - (w_{k+1}-w_k) = -(w_{k+1}-w_k) \rightarrow 0\end{aligned}$$
Finally, we want to show that there exists
$$d^k \in \partial_x \mathcal{L}_\beta(x_k,z_k,w_k) \quad \text{s.t} \quad \lim_{k\to\infty} d^k = 0.$$
Since $x^{k+1}$ is the minimum point of
$\Tilde{\mathcal{L}}^k_{\beta}(x,z_k,w_k)$, we have that
$0 \in \partial \Tilde{\mathcal{L}}^k_{\beta}(x,z_k,w_k)$. Using
Lemma [Lemma 7](#lemma_f1_f2){reference-type="ref"
reference="lemma_f1_f2"} and the definition of
$\Tilde{\mathcal{L}}^k_{\beta}$ we have: $$\begin{aligned}
    &\exists d_{k+1} \in \partial f(x_{k+1}) \\
    \label{equ:dk}
    s.t \quad & H^T w_k + L_x(x_{k+1}-x_k) + \beta H^T (Hx_{k} -z_{k}) + d_{k+1} = 0\end{aligned}$$
Lets us define:
$$\Tilde{d}_{k+1} = H^T w_{k+1} + \beta H^T (Hx_{k+1} -z_{k+1}) + d_{k+1}$$
we can easily verify that
$\Tilde{d}_{k+1} \in \partial_x \mathcal{L}_\beta(x_{k+1}, z_{k+1}, w_{k+1})$.
We arleady showed that the primal residues $\|x_{k+1}-x_k\|$,
$\|z_{k+1}-z_k\|$, $\|w_{k+1}-w_k\|$ converge to 0 as k approaches
infinity, therefore: $$\begin{aligned}
    \lim_{k\to\infty}  \Tilde{d}_{k+1} &  =  \lim_{k\to\infty} H^T w_{k+1} + \beta H^T (Hx_{k+1} -z_{k+1}) + d_{k+1}\\
    & = \lim_{k\to\infty} H^T w_k + L_x(x_{k+1}-x_k) + \beta H^T (Hx_{k} -z_{k}) + d_{k+1} = 0\end{aligned}$$
where the last equality is obtained
using [\[equ:dk\]](#equ:dk){reference-type="ref" reference="equ:dk"}.

# Application to PnP-LADMM

## Proof of Proposition 1

### Proximal Gradient Step Denoiser.

::: {.proof}
*Proof.* Let $\mathcal{D}_{\sigma_d}$ be the proximal gradient step
denoiser defined in [@Hurault2022] as
$\mathcal{D}_{\sigma_d}:= Id - \nabla g_{\sigma_d}$ where
$g_{\sigma_d}(x) = \frac{1}{2} \|x - N_{\sigma_d}\|^2$ and
$N_{\sigma_d}$ is a neural network.

According to [@Hurault2022 Proposition 3.1] there exists
$\phi_{\sigma_d}$ such that
$\mathcal{D}_{\sigma_d}= \ensuremath{\operatorname{prox}}_{\phi_{\sigma_d}}$.

In addition [@Hurault2022 Equation (26)] states that
$\phi_{\sigma_d}\geq g_{\sigma_d}$, and by definition
$g_{\sigma_d}\geq 0$.

Hence, $f=\phi_{\sigma_d}/ \sigma_d^2$ is lower bounded by 0 and
$\mathcal{D}_{\sigma_d}= \ensuremath{\operatorname{prox}}_{\sigma_d^2 f}$
as indicated by Proposition 1. ◻
:::

### MMSE Denoiser.

::: {.proof}
*Proof.* Let $\mathcal{D}_{\sigma_d}(y) = E[X|Y=y]$ be an MMSE denoiser,
where $Y=X+\sigma_d N$ and $N\sim \mathcal{N}(0,\sigma_d^2 Id)$, and
$X \sim p_X$, $p_X$ being a probability measure.

We want to show that there exists a lower bounded $\phi_{\sigma_d}$ such
that
$\mathcal{D}_{\sigma_d}(x)=\operatorname{prox}_{\phi_{\sigma_d}}(x)$.

For $\sigma_d=1$ according to [@Gribonval2011] there exists
$f(x) \geq - \log p_Y(x)$, such that
$\mathcal{D}_{1} = \operatorname{prox}_{f}$. $f$ is lower bounded
because the noisy density $p_Y(x) = (p_X * g_1) (x) \leq 1/\sqrt{2\pi}$
is upper-bounded by the maximum value of $g_1$ (the gaussian pdf with
identity covariance matrix).\
For $\sigma_d\neq 1$ the problem can be reduced to the previous case via
the following scaling: Consider
$\mathcal{P}(x) = \frac{1}{\sigma_d} \mathcal{D}_{\sigma_d}(\sigma_d x)$.
Then $\mathcal{P}(y) = E[\tilde{X}|\tilde{Y}=y]$ is an MMSE denoiser
with variance 1 with $\tilde{X} = X/\sigma_d$ and
$\tilde{Y}=\tilde{X} + N$. So we can find (according to the previous
argument for $\sigma_d=1$) $f$ such that
$\mathcal{P} = \operatorname{prox}_{f}$. Applying a change of variables
in the proximal operator we obtain
$$\mathcal{D}_{\sigma_d} (y) = \sigma_d \mathcal{P}(y/\sigma_d) = \operatorname{prox}_{\phi_{\sigma_d}}(y)$$
where $$\phi_{\sigma_d}(x) = \sigma_d^2 f(x/\sigma_d)$$ Finally, since
$f$ is lower-bounded $\phi_{\sigma_d}$ is lower bounded too. ◻
:::

# Convergence to critical point

Since we are optimizing $$E(x) = g(Hx) + f(x)$$ we would like to show
that $$\lim_{k\to\infty} \nabla E(x_k) = 0$$ We can almost conclude this
from Theorem 1. Indeed
$$\nabla E(x_k) = H^* \nabla g(Hx) + \nabla f(x)$$ From Theorem 1 we
have that $$\nabla_w {\mathcal L}_\beta = z_k - H x_k \to 0$$
$$\nabla_z {\mathcal L}_\beta = w_k + \beta(z - Hx) + \nabla g(z) \to 0$$
$$\nabla_x {\mathcal L}_\beta = \nabla f(x_k) - H^* w_k + \beta H^*(Hx_k - z_k) \to 0$$
Putting all together we have: $$\begin{aligned}
    z_k - H x_k & \to 0 \\
    \nabla g(z_k) + w_k & \to 0 \\
    \nabla f(x_k) - H^* w_k & \to 0\end{aligned}$$ Since $\nabla g$ is
continuous we get that
$$\nabla g(H x_k) + w_k \to \nabla g(z_k) + w_k \to 0$$ This means that
$$\begin{aligned}
    \nabla E(x_k) & = H^* \nabla g(H x_k) + \nabla f(x_k) \\
    & = H^* \nabla g(H x_k) + H^* w_k + \nabla f(x_k) - H^* w_k \\
    & = H^* (\nabla g(H x_k) + w_k) + (\nabla f(x_k) - H^* w_k) \\
    & \to 0 \;\end{aligned}$$ **Conclusion:** If $f$ is differentiable,
then Theorem 1 implies that the Linearized ADMM converges to a critical
point of the original objective $E(x)$.
