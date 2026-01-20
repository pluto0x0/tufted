#import "../index.typ": template, tufted
#show: template.with(title: "ELBO and VAE")
// #let note = tufted.margin-note
// #let note(t) = block(
//   fill: luma(230),
//   inset: 8pt,
//   radius: 4pt,
//   t
// )

#let NN = $cal(N)$
#let elbo = $cal(F)$
#let KL(x, y) = $"KL"(#x, #y)$

= ELBO

In Maximum Likelihood Estimation (MLE) our goal is to maximize the log likelihood of the parameter $theta$, which is defined as

$
ell(theta) = log p_theta (x)
= log sum_z p_theta (x, z)
$

where $x$ is the obervation and $z$ is the latent variable.
Computing $ell(theta)$ directly is often intractable due to the summation over latent variable $z$.
If we introduce an arbitrary distribution $q(z)$ over the latent variable $z$, we can derive a lower bound of $ell(theta)$ using Jensen's inequality:

$
ell(theta) &= log sum_z p_theta (x,z)
= log sum_z q(z) (p_theta (x,z)) / q(z)
= log EE_(z~q)[(p_theta (x,z)) / q(z)] \
&>= EE_(z~q) [log (p_theta (x,z)) / q(z)]
= EE_(z~q) [log p_theta (x,z)] - EE_(z~q) [log q(z)].
$

from which we define the Evidence Lower Bound (ELBO) $elbo$ as

$
ell(theta) >=
elbo(q, theta) equiv EE_(z~q) [log p_theta (x,z) - log q(z)].
$

Jensen's inequality becomes equality iff for all $z$,
$(p_theta (x,z)) / q(z)$ is a constant $c$. Solving for $q(z)$ gives

$
  p_theta (x) &= sum_z p_theta (x,z) = c sum_z q(z) = c \
  q(z) &= (p_theta (x,z)) / c = (p_theta (x,z)) / (p_theta (x)) = p_theta (z|x).
$

Note that $q(z)$ is an arbitrary distribution, but in practice we often choose $q(z)$ to be an *approximate* posterior distribution $q(z|x)$, and ELBO equals the log likelihood iff $q(z|x)$ is equal to the *true posterior* $p_theta (z|x)$.


== KL Divergence

Before diving deeper into ELBO, let's review some concepts in information theory.

The *entropy* of a distribution $p(x)$ is defined as

$
H(p) = EE_(x~p)[- log p(x)],
$

which is the expected amount of information needed to describe a random variable drawn from $p$.
And if we have another distribution $q(x)$ to approximate $p(x)$ and use the code length based on $q$, we define the *cross entropy* between $p$ and $q$ as

$
H(p, q) = EE_(x~p)[- log q(x)]
$

Finally, the KL divergence between $p$ and $q$ is defined as the difference between cross entropy and entropy:

$
KL(p, q) = H(p, q) - H(p) = EE_(x~p)[log p(x) - log q(x)].
$

KL divergence measures how different two distributions are.#footnote[However, KL divergence is not a true distance metric since it's not symmetric: $KL(p, q) != KL(q, p)$.] KL divergence is always non-negative:

$
KL(p, q)
=&EE_(x~p)[log q(x) / p(x)] \
=&-EE_(x~p)[log (p(x) / q(x))] \
>=& -log EE_(x~p)[q(x) / p(x)]  &&"(Jensen's Ineq.)"\
=& -log integral_x p(x) (q(x) / p(x)) dif x \
=& -log integral_x q(x) dif x \
=& -log 1 = 0
$

and thus $KL(p, q) = 0$ iff $p=q$.

Now we can rewrite ELBO in terms of KL divergence:

$
  elbo(q, theta)
   &= EE_(z~q) [log p_theta (x,z)] - EE_(z~q) [log q(z)] \
  &= EE_(z~q) [log (p_theta (x,z)) / (p_theta (z))] - (EE_(z~q) [log q(z)] + EE_(z~q) [log p_theta (z)]) \
  &= EE_(z~q) [log p_theta (x|z)] - KL(q(z), p_theta (z)).
$

In this form, computing ELBO only requires evaluating the conditional likelihood $p_theta (x|z)$ and the prior distribution $p_theta (z)$, which are often much easier than computing the marginal likelihood $p_theta (x)$.

== Gap Between Log Likelihood and ELBO

First we want to see the gapbetween log likelihood and ELBO:

$
ell(theta) - elbo(q, theta)
=& EE_(z~q)[log p_theta (x) - log p_theta (x,z) + log q(z)] \
=& EE_(z~q)[-log p_theta (z|x) + log q(z)] \
=& "KL"(q(z) || p_theta (z|x)) >= 0
$

It's easy to see that ELBO is a lower bound of log likelihood, and the gap is exactly the KL divergence between the approximate posterior $q(z)$ and the true posterior $p_theta (z|x)$, and the gap vanishes when the two posteriors are equal.

== EM algorithm and MLE

The target of MLE is to want to maximize the log likehood
$ell(theta) = log p_theta (x)$. We denote the $i$-th step 
of parameter as $theta^((i))$ and latent variable distribution as $q^((i))(z)$.

*E Step:*
In the E step, we set the latent variable distribution $q^((i))(z)$ to be the posterior given the current parameter $theta^((i))$:

$
  q^((i+1))(z) = p_(theta^((i))) (z|x)
$

*M Step:*
Once we have the latent variable distribution $q^((i+1))(z)$, we can update the model parameter $theta$ by maximizing the ELBO:

$
  theta^((i+1)) = arg max_theta elbo(q^((i+1)), theta)
$

*Monotonicity of Log Likelihood:*
We can show that the log likelihood is non-decreasing after each EM step:

$
  &ell(theta^((i+1))) \
  >=& elbo(q^((i+1)), theta^((i+1))) && "(lower bound)" \
  >=& elbo(q^((i+1)), theta^((i))) && (arg max)\
  =& ell(theta^((i))) && (q^((i+1)) = p_(theta^((i))) (z|x))
$

// *GMM.*
// Assume observed data $x$ comes from a mixture of $K$ Gaussian distributions:
// $p_theta (x) = sum_(k=1)^K pi_k NN(x|mu_k, Sigma_k)$.
// where
// $theta = {pi_k, mu_k, Sigma_k}_(k=1)^K$ are model parameters, $pi_k$ are mixture coefficients satisfying $sum_(k=1)^K pi_k = 1$.
// *E Step.*
// Compute posterior probabilities (responsibilities):
// $gamma_(z^(i)=k) &<- p_(theta) (z^(i)=k|x^(i)) = (pi_k NN(x^(i)|mu_k, Sigma_k)) / (sum_(j=1)^K pi_j NN(x^(i)|mu_j, Sigma_j))$.
// *M Step.*
// Update parameters:
// $theta = arg max_theta elbo(gamma, theta)$
// i.e. maximize likelihood given responsibilities $gamma_(z^(i)=k)$.
// Define $N_k & = sum_(i=1)^N gamma_(z^(i)=k)$, then
// $pi_k & <- N_k / N space\;
// mu_k & <- (1 / N_k) sum_(i=1)^N gamma_(z^(i)=k) x^(i) space\;
// Sigma_k & <- (1 / N_k) sum_(i=1)^N gamma_(z^(i)=k) (x^(i) - mu_k)(x^(i) - mu_k)^top$

// == 例子：高斯混合模型 GMM

// 假设观测数据 $x$ 来自 $K$ 个高斯分布的混合：

// $
//   p_theta (x) = sum_(k=1)^K pi_k NN(x|mu_k, Sigma_k)
// $

// 其中 $theta = {pi_k, mu_k, Sigma_k}_(k=1)^K$ 是模型参数，$pi_k$ 是混合系数，满足 $sum_(k=1)^K pi_k = 1$.
// 数据样本为 ${x^(i)}_(i=1)^N$，引入隐变量 $z$ 表示样本属于哪个高斯分布, $z^(i) = k$ 表示样本 $x^(i)$ 来自第 $k$ 个高斯分布.

// === E 步

// 计算后验概率（责任度）：

// $
//   gamma_(z^(i)=k) &<- p_(theta) (z^(i)=k|x^(i)) \
//   &= (pi_k NN(x^(i)|mu_k, Sigma_k)) / (sum_(j=1)^K pi_j NN(x^(i)|mu_j, Sigma_j))
// $

// === M 步

// 更新参数：

// $
//   theta = arg max _theta cal(F)(gamma, theta)
// $

// 即在已知责任度 $gamma_(z^(i)=k)$ 下，最大化似然的参数. 定义 $N_k & = sum_(i=1)^N gamma_(z^(i)=k)$,

// $
//   pi_k &<- N_k / N \
//   mu_k &<- (1 / N_k) sum_(i=1)^N gamma_(z^(i)=k) x^(i) \
//   Sigma_k &<- (1 / N_k) sum_(i=1)^N gamma_(z^(i)=k) (x^(i) - mu_k)(x^(i) - mu_k)^top \
// $

= VAE

A common approach in generative modeling is Autoencoder, which maps data $x$ to a latent variable $z$ using an encoder, and then reconstructs $x$ from $z$ using a decoder.

Based on that, the Variational Autoencoder (VAE) treats the latent variable $z$ as a random variable and have the encoder predict a distribution over $z$ given $x$. With this probabilitic interpretation, it becomes possible to sample new data by sampli ng $z$ from a prior distribution, and also other operations like interpolation in the latent space.

In VAE, *encoder* is the approximate posterior $q_phi (z|x)$, *decoder* is the generative distribution $p_theta (x|z)$, and *prior* distribution of latent variable is $p(z)$ which is usually assigned to standard Gaussian distribution $NN(0, I)$.


// #note([
//   image
// ])


== Loss Function

We use ELBO as approximation of log likelihood:

$
  cal(L)
  = -elbo(q_phi, theta)
  = -EE_(q_phi (z|x)) [log p_theta (x|z)] + KL(q_phi (z|x), p(z))
$

where $EE_(q_phi (z|x)) [log p_theta (x|z)]$ is the *reconstruction term*, which encourages the decoder to reconstruct input $x$ from latent variable $z$ accurately;
$KL(q_phi (z|x), p(z))$ is the *regularization term*, encouraging the approximate posterior distribution $q_phi (z|x)$ to be close to the prior distribution $p(z)$.

== Training and Reparameterization

In practice, the exceptation in the reconstruction term is often intractable, and we use Monte Carlo sampling to approximate it. However, naively sampling $z$ from $q_phi (z|x)$ would break the computational graph and prevent gradients from flowing back to the encoder parameters $phi$. To address the sampling $z ~ NN(mu(x), sigma^2(x))$, we use the reparameterization trick:

$
  z = mu(x) + sigma(x) dot.o epsilon, epsilon ~ NN(0, I).
$

// *Breaking Down the KL Term.*
// Define $q_"agg" (z) = EE_(x~P(x)) q(z|x)$, then $EE_(x~p_"data") "ELBO"$ is
// $underbrace(EE_(x~p_"data") EE_(q_phi (z|x)) [log p_theta (x|z)], "Reconstruction")
// - underbrace(H(q_"agg" (z), p(z)), "Cross Entropy")
// + underbrace(H(p(z)), "Entropy")$
// where they encourages the latent distribution $q_"agg"$ to:
// Reconstruction: deviate from the prior distribution $p(z)$, improving reconstruction quality.
// Cross Entropy: pull towards the center of the prior distribution $p(z)$, reducing both mean and variance.
// Entropy: increase variance, making the it flatter.

== Problems in VAE

*Prior Hole:*
Define aggregated latent distribution as $q_"agg" (z) = EE_(x~P(x)) q(z|x)$, then if $q_"agg" (z)$ does not cover the whole pior $p(z)$, it leaves holes that $q_"agg" (z)$ never visits and consequently poor reconstruction quality.

*Posterior Collapse:*
If the decoder is too powerful e.g. autoregressive models, it can ignore the latent variable $z$ entirely and reconstruct $x$ directly from its learned parameters. In this case, the approximate posterior $q_phi (z|x)$ collapses to the prior $p(z)$, leading to ineffective latent representations.

$
  exists i space s.t. space forall x, q_phi (z_i|x) approx p(z_i)
$

== Vector Quantized VAE(VQ VAE)

// Let $z$ be discrete latent variable. VQ VAE defines a discrete codebook and maps the continuous encoder output to the nearest codebook entry. This solves the Prior Hole and Posterior Collapse issues in standard VAE but requires special training techniques to handle the non-differentiability of discrete variables.
// Formally,

In VQ VAE, the latent variable $z$ is discrete and represented by a codebook of embeddings ${e_k}_(k=1)^K$. The encoder outputs a continuous representation $z_e (x)$, which is then quantized to the nearest codebook entry:

$
  q(z=k|x) = bb(1)[k=arg min_j norm(z_e (x) - e_j)_2]
$
