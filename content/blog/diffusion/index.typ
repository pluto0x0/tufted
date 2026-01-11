// #import "../index.typ": template, tufted
// #show: template.with(title: "Diffusion")

// #set heading(numbering: "1.")
// #outline()
// #set math.equation(numbering: "a.")

#let NN = $cal(N)$
#let LL = $cal(L)$
#let balpha = $overline(alpha)$
#let KL(x, y) = $"KL"(#x||#y)$

= Diffusion Models

The diffusion models are generative models that learn to generate data by reversing a gradual noising process. The idea is an analogue of diffusion in physics.

== Forward noising (Markov chain)

First we start from a data sample from the data distribution $x_0 ~ q(x)$. Define a Markov chain ${x_i}_0^T$ and a forward noising process

$
  q(x_t|x_(t-1)) = NN(x_t\; sqrt(1 - beta_t) x_(t-1), beta_t I)
$

where $beta_t$ is a small positive variance schedule.
With the reparameterize trick, the forward process is equivalently
$x_t = sqrt(1 - beta_t) x_(t-1) + sqrt(beta_t) epsilon_t$, $epsilon_t ~ NN(0, I)$.

Let $alpha_t = 1 - beta_t$ and $balpha_t = product_(s=1)^t alpha_s$, then we have

$
  x_t &= sqrt(alpha_t) x_(t-1) + sqrt(1 - alpha_t) epsilon_t \
      &= sqrt(alpha_t alpha_(t-1)) x_(t-2) + sqrt(alpha_t (1 - alpha_(t-1))) epsilon_t + sqrt(1 - alpha_t) epsilon_t \
      &= sqrt(alpha_t alpha_(t-1)) x_(t-2) + sqrt(1 - alpha_t alpha_(t-1)) epsilon \
      &= sqrt(alpha_t alpha_(t-1) alpha_(t-2)) x_(t-3) + sqrt(1 - alpha_t alpha_(t-1) alpha_(t-2)) epsilon \
      &= dots \
      &= sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon, quad epsilon ~ N(0, I).

$

i.e.

$
  q(x_t | x_0) = NN(x_t\; sqrt(balpha_t) x_0, (1 - balpha_t) I).
$

Note that as $t -> infinity$, $balpha_t -> 0$ and $x_t ~ NN(0, I)$, which means the data is gradually noised to pure Gaussian noise.

== Reverse denoising
True reverse $q(x_(t-1) | x_t)$ is intractable, approximate with
$p_theta (x_(t-1) | x_t) = NN(x_(t-1)\; mu_theta (x_t, t), Sigma_theta (x_t, t))$.

*ELBO Loss.*
Negative log-likelihood $- log p_theta(x_0)$ is minimized via ELBO with latent $x_(1:T)$ and obervation $x_0$. By Markov property,
$q(x_(1:T)|x_0) = q(x_T | x_0) product_(t=2)^T q(x_(t-1) | x_t, bold(x_0))$ and
$p_theta (x_0:T) = p_theta (x_T) product_(t=1)^T p_theta (x_(t-1) | x_t)$.
Using factorization, ELBO reduces to
$=& EE_q(x_1:x_T|x_0) [log p_theta (x_T) + sum_(t=1)^T log p_theta (x_(t-1) | x_t) - log q(x_T | x_0) - sum_(t=2)^T q(x_(t-1) | x_t,x_0)] \
=& EE_q(x_1|x_0) [log p_theta (x_0|x_1)]
- sum_(t=2)^T EE_q(x_t,x_(t-1)|x_0) [log q(x_(t-1) | x_t,x_0) - log p_theta (x_(t-1) | x_t)]
+ EE_q(x_T|x_0) [log p_theta (x_T) - log q(x_T | x_0)] \
=& EE_q(x_1|x_0) [log p_theta (x_0|x_1)]
- sum_(t=2)^T EE_q(x_t|x_0) EE_q(x_(t-1)|x_t,x_0) [log q(x_(t-1) | x_t,x_0) - log p_theta (x_(t-1) | x_t)]
+ EE_q(x_T|x_0) [log p_theta (x_T) - log q(x_T | x_0)] \
=& EE_q(x_1|x_0) [log p_theta (x_0|x_1)]
- sum_(t=2)^T EE_q(x_t|x_0) KL(q(x_(t-1) | x_t,x_0), p_theta (x_(t-1) | x_t))
- KL(log q(x_T | x_0), log p_theta (x_T))$
or equivalently,
$LL = -cal(F)(q, theta) = LL_0 + sum_(t=2)^T LL_(t-1) + LL_T$, where
$LL_0 = -EE_q(x_1|x_0)[log p_theta (x_0|x_1)]$ is the reconstruction term,
$LL_(t-1) = EE_q(x_t|x_0) KL(q(x_(t-1)|x_t,x_0), p_theta (x_(t-1)|x_t))$ is the denoising matching term, and
$LL_T = KL(q(x_T|x_0), p_theta (x_T))$ is the prior matching term,
which $approx 0$ for large $T$ because both are close to $NN(0, I)$.
With Gaussian parameterization, $LL_0$ corresponds to MSE reconstruction loss.

*One step Denoise.*
The *multiplication* of 2 Gaussians is,
$NN(x\; mu_1, Sigma_1) NN(x\; mu_2, Sigma_2)
prop NN(x\; mu, Sigma)$
where
$Sigma = (Sigma_1^(-1) + Sigma_2^(-1))^(-1)
, mu = Sigma (Sigma_1^(-1) mu_1 + Sigma_2^(-1) mu_2)$
Consider *one step denoising*:
$q(x_(t-1)|x_t,x_0) = (q(x_t | x_(t-1), x_0) q(x_(t-1)|x_0)) / q(x_t|x_0)
prop NN(x_t\; sqrt(alpha_t) x_(t-1), (1-alpha_t) I)
NN(x_(t-1)\; sqrt(balpha_(t-1)) x_0, (1 - balpha_(t-1)) I)
prop NN(x_(t-1)\; (sqrt(alpha_t)(1-balpha_(t-1))x_t + sqrt(balpha_(t-1))beta_t x_0)/(1-balpha_t), ((1-alpha_t)(1-balpha_(t-1)))/(1-balpha_t)I)$
because
$NN(x_t\; sqrt(alpha_t) x_(t-1), beta_t I) prop NN(x_(t-1)\; (1 / sqrt(alpha_t)) x_t, (beta_t / alpha_t) I)$.
Recall $NN(x, mu, Sigma) = 1/(sqrt((2 pi)^d abs(Sigma))) exp(-1/2 (x - mu)^T Sigma^(-1) (x - mu))$.
This is exactly the target of $p_theta (x_(t-1) | x_t)$, measurred by KL div in $LL_(t-1)$.
We fount that variance $Sigma_q (t)$ is independent of $x$, only need to learn mean $mu_theta$.
Recall the *KL div. between 2 Gaussians*:
$KL(NN(mu_1, Sigma_1), NN(mu_2, Sigma_2))
= 1/2 [log (abs(Sigma_2) / abs(Sigma_1)) - d + tr(Sigma_2^(-1) Sigma_1) + (mu_2 - mu_1)^T Sigma_2^(-1) (mu_2 - mu_1)]$.
Since the variance are the same, minimizing $LL_(t-1)$ is equivalent to minimizing
$EE_q(x_t|x_0) [1/(2 sigma_q^2(t))norm(mu_q (x_t, x_0, t) - mu_theta (x_t, t))_2^2]$
The model predicts $hat(x_theta)(x_t, t)$, so our predicted mean is $mu_theta (x_t, t)=mu_q (x_t, hat(x_theta)(x_t, t), t)$. Plug in, the loss is $EE_q(x_t|x_0) [1/(2 sigma_q^2(t)) (balpha_(t-1) beta_t^2)/((1-balpha_t)^2) norm(hat(x_theta)(x_t, t) - x_0)_2^2]$.

*Training.*
Steps: (1) sample $x_0 ~ q(x)$; (2) sample $t ~ "Unif"({1,...,T})$; (3) sample $epsilon ~ NN(0, I)$; (4) compute $x_t = sqrt(balpha_t) x_0 + sqrt(1 - balpha_t) epsilon$; (5) predict $hat(x_theta)(x_t, t)$; (6) compute loss $cal(L)_t = w(t) norm(hat(x_theta)(x_t, t) - x_0)_2^2$ where $w(t) = 1/(2 sigma_q^2(t)) (balpha_(t-1) beta_t^2)/((1-balpha_t)^2)$; (7) update theta by minimizing $cal(L)_t$.

*Predicting the Noise.*
Rewrite $x_0 = (x_t - sqrt((1-balpha_t)) epsilon_0)/(sqrt(balpha_t))$ where $epsilon_0 ~ NN(0, I)$. Now predict $epsilon_0$ instead of $x_0$. The loss becomes $EE_q(x_t|x_0) [1/(2 sigma_q^2(t)) (1-alpha_t)^2/((1-balpha_t)alpha_t) norm(hat(epsilon_theta)(x_t, t) - epsilon_0)_2^2]$.

= Score-Based Diffusion

*Score function.*
Define $"score"(x) = nabla_x log p(x)$.
*Tweedie’s formula.*
If $z ~ NN(mu_z, Sigma_z)$, then $EE[mu_z|z] = z + Sigma_z nabla_z log p(z)$.
Recall $q(x_t | x_0) = NN(x_t\; sqrt(balpha_t) x_0, (1 - balpha_t) I)$ is a Gaussian. So we can reparameterize the $x_0$ as
$x_0 = (x_t + (1 - balpha_t) nabla_(x_t) log p(x_t)) / sqrt(balpha_t)$.
Thus predicting a denoiser is equivalent to predicting score. Write the loss
$EE_q(x_t|x_0) [1/(2 sigma_q^2(t)) (1-alpha_t)^2/alpha_t [norm(s_theta (x_t, t) - nabla log p(x_t))_2^2]$.

*Langevin sampling (discrete).*
Given score of $pi(x)$, iterate
$X_(k+1) = X_k + tau nabla_x log pi(X_k) + sqrt(2 tau) xi$, $xi ~ NN(0, I)$,
then $X_k ~ pi(x)$ as $k -> infinity$.
This is noisy gradient ascent toward high-density regions.
*Connection to diffusion.*
Reverse-time denoising in diffusion models takes the same form, with learned score
$nabla_x log p_t(x)$ replacing the true score.
*Simulated Annealing.* by choosing decreasing noise levels, the sample converges to high-density modes of the data distribution.

= Conditional Diffusion
*Conditional score.*
Target $nabla_x log p(x|c) = nabla_x log p(x) + nabla_x log p(c|x)$.
*Classifier Guidance.* Use a classifier $h$ on noised $x_t$. The prediction probability is $"softmax"(p_i (x_t)) := (exp h_i (x_t)) / (sum_j exp h_j (x_t))$. Then $nabla_x log p(c|x)$ is $nabla_x h_c (x) - nabla_x log (sum_j exp h_j (x))$. Use backprop to compute gradients, or just use $nabla_x h_c (x)$. Further, define $nabla_x log p(x|c) = nabla_x log p(x) + s nabla_x log p(c|x)$, scale by $s > 1$ to strengthen conditioning.

*Classifier Free Guidance (CFG).*
Define a Conditional Denoiser $D_theta (x_t, sigma, c) -> hat(x_0)$ that takes condition $c$ as input.
*Two scores.*
Conditional: $nabla_x log p_theta(x_t, c) = (D_theta (x_t, sigma, c) - x_t) / sigma^2$.
Unconditional: $nabla_x log p_theta(x_t) = (D_theta (x_t, sigma, emptyset) - x_t) / sigma^2$.
where $emptyset$ denotes no condition.
*CFG combination.*
$nabla_x log p(x|c) = S nabla_x log p_theta (x_t, c) + (1 - S) nabla_x log p_theta (x_t)$
$= 1/sigma^2 (S D_theta (x_t, sigma, c) + (1 - S) D_theta (x_t, sigma, emptyset) - x_t)$.
