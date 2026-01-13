#import "../index.typ": template, tufted
#show: template.with(title: "Diffusion")

// #set heading(numbering: "1.")
// #outline()
// #set math.equation(numbering: "a.")

#let NN = $cal(N)$
#let LL = $cal(L)$
#let balpha = $overline(alpha)$
#let KL(x, y) = $"KL"(#x||#y)$
#let elbo = $cal(F)$

= Diffusion Models

The diffusion models are generative models that learn to generate data by reversing a gradual noising process. The idea is an analogue of diffusion in physics.

== Forward noising (Markov chain)

First we start from a data sample from the data distribution $x_0 ~ q(x)$. Define a Markov chain ${x_i}_0^T$ and a forward noising process

$
  q(x_t|x_(t-1)) = NN(x_t\; sqrt(1 - beta_t) x_(t-1), beta_t I)
$

where $beta_t$ is a small positive variance schedule which satisfies $0 < beta_1 < beta_2 < ... < beta_T < 1$.
With the reparameterize trick, the forward process is equivalently
$x_t = sqrt(1 - beta_t) x_(t-1) + sqrt(beta_t) epsilon_t$, $epsilon_t ~ NN(0, I)$.

Let $alpha_t = 1 - beta_t$ and $balpha_t = product_(s=1)^t alpha_s$, then we have

$
  x_t & = sqrt(alpha_t) x_(t-1) + sqrt(1 - alpha_t) epsilon_t \
      & = sqrt(alpha_t alpha_(t-1)) x_(t-2) + sqrt(alpha_t (1 - alpha_(t-1))) epsilon_t + sqrt(1 - alpha_t) epsilon_t \
      & = sqrt(alpha_t alpha_(t-1)) x_(t-2) + sqrt(1 - alpha_t alpha_(t-1)) epsilon \
      & = sqrt(alpha_t alpha_(t-1) alpha_(t-2)) x_(t-3) + sqrt(1 - alpha_t alpha_(t-1) alpha_(t-2)) epsilon \
      & = dots \
      & = sqrt(overline(alpha)_t) x_0 + sqrt(1 - overline(alpha)_t) epsilon, quad epsilon ~ N(0, I).
$

i.e.

$
  q(x_t | x_0) = NN(x_t\; sqrt(balpha_t) x_0, (1 - balpha_t) I).
$

Note that as $t -> infinity$, $balpha_t -> 0$ and $x_t ~ NN(0, I)$, which means the data is gradually noised to pure Gaussian noise.

== Reverse denoising

Obviously, it is impossible to find the true reverse process $q(x_(t-1) | x_t)$, so we approximate with

$
  p_theta (x_(t-1) | x_t) = NN(x_(t-1)\; mu_theta (x_t, t), Sigma_theta (x_t, t)).
$

== ELBO Loss

The loss function is defined as negative log-likelihood $-log p_theta (x_0)$, which is minimized via ELBO $elbo(q(x_(1:T)), theta)$ with the $x_0$ as observation and $x_(1:T)$ as the *latent variables*. Then by Markov property, we have the latent variable posterior

$ q(x_(1:T)|x_0) = q(x_T | x_0) product_(t=2)^T q(x_(t-1) | x_t, bold(x_0)) $ and

$
  p_theta (x_0:T) = p_theta (x_T) product_(t=1)^T p_theta (x_(t-1) | x_t)
$

Substituting into ELBO, we have

$
  &EE_q(x_1:x_T|x_0) [log p_theta (x_0:T) - log q(x_1:T|x_0)] \
  =& EE_q(x_1:x_T|x_0) [log p_theta (x_T) + sum_(t=1)^T log p_theta (x_(t-1) | x_t) - log q(x_T | x_0) - sum_(t=2)^T q(x_(t-1) | x_t,x_0)] \
  =& EE_q(x_1|x_0) [log p_theta (x_0|x_1)]
  - sum_(t=2)^T EE_q(x_t,x_(t-1)|x_0) [log q(x_(t-1) | x_t,x_0) - log p_theta (x_(t-1) | x_t)] \
  & + EE_q(x_T|x_0) [log p_theta (x_T) - log q(x_T | x_0)] \
  =& EE_q(x_1|x_0) [log p_theta (x_0|x_1)]
  - sum_(t=2)^T EE_q(x_t|x_0) EE_q(x_(t-1)|x_t,x_0) [log q(x_(t-1) | x_t,x_0) - log p_theta (x_(t-1) | x_t)] \
  &+ EE_q(x_T|x_0) [log p_theta (x_T) - log q(x_T | x_0)] \
  =& EE_q(x_1|x_0) [log p_theta (x_0|x_1)]
  - sum_(t=2)^T EE_q(x_t|x_0) KL(q(x_(t-1) | x_t,x_0), p_theta (x_(t-1) | x_t)) \
  &- KL(log q(x_T | x_0), log p_theta (x_T))
$

we can further denote the result above as

$
  LL = -cal(F)(q, theta) = LL_0 + sum_(t=2)^T LL_(t-1) + LL_T
$

where

- $LL_0 = -EE_q(x_1|x_0)[log p_theta (x_0|x_1)]$ is the reconstruction term of the last step $x_1$ to $x_0$. #footnote[When distribution $p_theta (x_0|x_1)$ is Gaussian, this term is equivalent to a MSE loss between $x_0$ and the predicted mean from $x_1$: $EE[norm(x_1 - x_0)^2]$]
- $LL_(t-1) = EE_q(x_t|x_0) KL(q(x_(t-1)|x_t,x_0), p_theta (x_(t-1)|x_t))$ is the denoising matching term
- $LL_T = KL(q(x_T|x_0), p_theta (x_T))$ is the prior matching term, which is close to 0 for large $T$ because both distributions are close to $NN(0, I)$. This term is often ignored in practice.

== One step Denoise

Consider one-step denoising term $LL_(t-1)$. The true posterior $q(x_(t-1)|x_t,x_0)$ is the learning target of our denoiser model $p_theta (x_(t-1) | x_t)$, which can be derived by Bayes' rule:
#footnote[
  The multiplication of 2 Gaussians cdf is
  $
    NN(x\; mu_1, Sigma_1) NN(x\; mu_2, Sigma_2) prop NN(x\; mu, Sigma)
  $
  where $Sigma = (Sigma_1^(-1) + Sigma_2^(-1))^(-1)
  , mu = Sigma (Sigma_1^(-1) mu_1 + Sigma_2^(-1) mu_2)$
]
#footnote[
  By the symmetry of Gaussian cdf between $x$ and $mu$, we have
  $
    NN(x_t\; sqrt(alpha_t) x_(t-1), beta_t I) prop NN(x_(t-1)\; (1 / sqrt(alpha_t)) x_t, (beta_t / alpha_t) I).
  $
  // Recall $NN(x, mu, Sigma) = 1/(sqrt((2 pi)^d abs(Sigma))) exp(-1/2 (x - mu)^T Sigma^(-1) (x - mu))$.
]

$
  q(x_(t-1)|x_t,x_0) &= (q(x_t | x_(t-1), x_0) q(x_(t-1)|x_0)) / q(x_t|x_0) \
  &prop NN(x_t\; sqrt(alpha_t) x_(t-1), (1-alpha_t) I)
  NN(x_(t-1)\; sqrt(balpha_(t-1)) x_0, (1 - balpha_(t-1)) I) \
  &prop NN(x_(t-1)\; (sqrt(alpha_t)(1-balpha_(t-1))x_t + sqrt(balpha_(t-1))beta_t x_0)/(1-balpha_t), ((1-alpha_t)(1-balpha_(t-1)))/(1-balpha_t)I)
$

For simplicity, we denote

$
  q(x_(t-1)|x_t,x_0) =: NN(x_(t-1)\; mu_q (x_t, x_0, t), sigma_q (t)^2 I)
$

We can find out that the variance $sigma_q (t) I$ is independent of $x$, which means we only need to learn mean $mu_theta$. Furthermore, if the model predicts the original data $x_0$ as $hat(x_theta)(x_t, t)$, then we can obtain the predicted mean as $mu_theta (x_t, t)=mu_q (x_t, hat(x_theta)(x_t, t), t)$.

Therefore, by decomposing
#footnote[
  KL divergence between 2 Gaussians is:
  $
    KL(NN(mu_1, Sigma_1), NN(mu_2, Sigma_2))
    = 1/2 [log (abs(Sigma_2) / abs(Sigma_1)) - d + tr(Sigma_2^(-1) Sigma_1) + (mu_2 - mu_1)^T Sigma_2^(-1) (mu_2 - mu_1)].
  $
]
KL divergence term, minimizing $LL_(t-1)$ is equivalent to minimizing


$
  EE_q(x_t|x_0) [1/(2 sigma_q^2(t))norm(mu_q (x_t, x_0, t) - mu_theta (x_t, t))_2^2]
$

By eliminating $mu_theta (x_t, t)$ in, the loss becomse

$
  EE_q(x_t|x_0) [1/(2 sigma_q^2(t)) (balpha_(t-1) beta_t^2)/((1-balpha_t)^2) norm(hat(x_theta)(x_t, t) - x_0)_2^2].
$<eq:denoise-loss-x0>

== Training

Now we can summarize the training procedure as follows:

+ sample $x_0 ~ q(x)$
+ sample $t ~ "Unif"({1,...,T})$
+ sample $epsilon ~ NN(0, I)$
+ compute $x_t = sqrt(balpha_t) x_0 + sqrt(1 - balpha_t) epsilon$
+ predict $hat(x_theta)(x_t, t)$
+ compute loss $cal(L)_t = w(t) norm(hat(x_theta)(x_t, t) - x_0)_2^2$ where $w(t) = 1/(2 sigma_q^2(t)) (balpha_(t-1) beta_t^2)/((1-balpha_t)^2)$
+ update $theta$ by minimizing $cal(L)_t$

== Prediction Over Other Terms

We've shown how to predict the mean $mu_theta (x_t, t)$ by predicting $x_0$. However, recall the closed form of $x_t$ in terms of $x_0$ and $epsilon$:

$
  x_t = sqrt(balpha_t) x_0 + sqrt(1 - balpha_t) epsilon, quad epsilon ~ NN(0, I).
$

We can rewrite it as $x_0 = (x_t - sqrt((1-balpha_t)) epsilon)/(sqrt(balpha_t))$ 
Therefore, if we predict $epsilon_0$ instead of $x_0$, the loss becomes

$
  EE_q(x_t|x_0) [1/(2 sigma_q^2(t)) (1-alpha_t)^2/((1-balpha_t)alpha_t) norm(hat(epsilon_theta)(x_t, t) - epsilon_0)_2^2].
$

Compared to predicting $x_0$, $epsilon_0$ follows a standard Gaussian distribution, thus has better numerical stability.

In fact, we can predict any linear combination of $x_0$ and $epsilon_0$, e.g. in the v-parameterization, we instead predict

$
  // x_t = sqrt(balpha_t) x_0 + sqrt(1 - balpha_t) epsilon
  v = sqrt(balpha_t) epsilon - sqrt(1 - balpha_t) x_0
$

#tufted.margin-note(
  image("img/image.png", width: 75%)
)

#tufted.margin-note[
  Illustration of $x_t$ and $v$ in the $x_0$-$epsilon$ space
]

Note that the coefficients of $x_0$ and $epsilon$ in $x_t$ are both positive, while $(sqrt(balpha_t)^2 + sqrt(1 - balpha_t)^2) = 1$. Therefore, the $x_t$ is a unit vector in the $x_0$-$epsilon$ space, while $v$ is the speed vector orthogonal to $x_t$.


= Score-Based Diffusion


We first define *score function*, which is the gradient of log probability density,

$
  s(x) = nabla_x log p(x).
$

*Tweedie's formula:*

If $z ~ NN(mu_z, Sigma_z)$, then

$
  EE[mu_z|z] = z + Sigma_z nabla_z log p(z)
$.

Recall $q(x_t|x_0) = NN(x_t\; sqrt(balpha_t) x_0, (1 - balpha_t) I)$. So we can reparameterize the $x_0$ with Tweedie's formula as

$
  x_0 = (x_t + (1 - balpha_t) nabla_(x_t) log p(x_t)) / sqrt(balpha_t).
$

i.e. the score function is lienarly related to $x_t$ and $x_0$,
thus score predictions can be equivalently used as a denoiser.
Substituting into the denoising loss @eq:denoise-loss-x0, we have

$
  EE_q(x_t|x_0) [1/(2 sigma_q^2(t)) (1-alpha_t)^2/alpha_t norm(s_theta (x_t, t) - nabla log p(x_t))_2^2].
$<eq:score-loss>

== Langevin sampling (discrete)

The discrete Langevin equation shows that, given score of $pi(x)$, and a step size $tau > 0$, the Markov chain

$
  X_(k+1) = X_k + tau nabla_x log pi(X_k) + sqrt(2 tau) xi, quad xi ~ NN(0, I),
$

then $X_k ~ pi(x)$ as $k -> infinity$.
This process is a noisy gradient ascent toward high-density regions, which enables sampling from complex distributions with only the need of score function.

#tufted.margin-note[
  In statistical mechanics, the Boltzmann distribution states that the distribution over system states is

  $
    p_i prop exp(-epsilon_i / (k T))
  $ <eq:boltzmann>

  where $epsilon_i$ is the energy of state $i$, $k$ is the Boltzmann constant, and $T$ is the temperature.

  The Langevin equation describes the Brownian motion of a particle in a (one-dimensional) potential field,

  $
    dif x_t = -1/gamma nabla_x U(x_t) dif t + sqrt((2 k T) / gamma) dif W_t
  $ <eq:langevin>

  where

  - $x$ is the particle position
  - $U(x)$ is the potential energy function
  - $gamma$ is the damping coefficient
  - $W_t$ is a standard Wiener process: $W_(t+Delta) = W_t + NN(0, Delta)$

  According to the Boltzmann distribution,

  $
    U(x) = - k T log p(x) + "constant",
  $

  substituting gives

  $
    dif x_t &= (k T)/gamma nabla_x log p(x_t) dif t + sqrt((2 k T) / gamma) dif W_t \
    &= (k T)/gamma nabla_x log p(x_t) dif t + sqrt((2 k T) / gamma) dif W_t
  $

  In discrete time, with $x_k := x(k tau)$, the equation becomes

  $
    x_(k+1) - x_k &= - (k T)/gamma tau nabla_x log p(x_t) + sqrt((2k T)/gamma tau) xi quad &, xi ~ NN(0, I) \
    &= - eta nabla_x log p(x_t) + sqrt(2 eta) xi &, xi ~ NN(0, I)
  $

  where $eta = (k T)/gamma tau$ is the step size. Recall that $x_t$ denotes the random position of the particle, thus

  $
    x_k ~ p(x)
  $
]

#tufted.margin-note[
  If we keep the potential energy in the Boltzmann distribution fixed and let
  $T$ gradually converge to $0$, the distribution converges to a point mass
  $p(x) = delta(x - x^*)$, where $x^* = arg min_{x} U(x)$ is the point of
  minimum potential energy. In this case, the stochastic perturbation term in
  the Langevin equation converges to $0$, and the Langevin dynamics degenerates
  into gradient descent on the potential. The rate at which the temperature $T$
  is decreased controls the randomness of the sampling process: a fast cooling
  schedule leads to insufficient exploration and getting stuck in local minima
  of the potential, whereas a slower schedule gives a higher probability of
  reaching the global minimum. This method of simulating the cooling process is
  known as *Simulated Annealing*.
]

// *Connection to diffusion.*
// Reverse-time denoising in diffusion models takes the same form, with learned score
// $nabla_x log p_t(x)$ replacing the true score.
// *Simulated Annealing.* by choosing decreasing noise levels, the sample converges to high-density modes of the data distribution.

== Learning Score Function

We've shown in the previous section that denoising is equivalent to learning the score function

$
  s_theta (x_t, t) approx nabla_(x_t) log p_t (x_t),
$

and now we will see how to directly learn the score function. Since the data distribution $p(x)$ is unknown, we instead learn the score of the noised data distribution $q(x_t|x_0)$ at different noise levels $t$. We can assume the data distribution is Gaussian, i.e. $q(x_t|x_0) = NN(x_t\; x_0, sigma_t^2 I)$, then we can write the score function. Denote a sample $u = x + sigma_t z$ with $z ~ NN(0, I)$, then

$
  nabla_u log NN(u\; x, sigma_t^2 I)
  = nabla_u (-(u-x)^top (u-x)) / (2 sigma_t^2)
  = -(u - x) / sigma_t^2
  = z / sigma_t.
$

We can thus define the Denoising Score Matching (DSM) loss as

$
  LL_"DSM" = EE_(x,z,t) [lambda(t) norm(s_theta (x+sigma_t z, t) + z / sigma_t)_2^2]
$

where $lambda(t)$ is a weighting function which balances the loss at different noise levels . A common choice is to normalize $lambda(t) (1/sigma_t)^2$, i.e. $lambda(t) = sigma_t^2$.

// Comparing the denoising loss with score prediction in @eq:score-loss and DSM loss, we can see they are equivalent up to a constant factor

// and if we want to directly learn the score function, we can sample a data $x_0$ and learn the posterior score $nabla_(x_t) log q(x_t|x_0)$ via minimizing

= Conditional Diffusion

In pratice, we often want to generate data conditioned on some information $c$, e.g. class labels or text descriptions. We can achieve this by learning the *conditional score function:*

$
  nabla_x log p(x|c) = nabla_x log p(x) + nabla_x log p(c|x).
$

Since we've learned $nabla_x log p(x)$ in the unconditional diffusion model, we only need to find a way to compute $nabla_x log p(c|x)$.

== Classifier Guidance

A simple way is to train a classifier $p_i (x) = p(c=i|x)$ on the noised data $x_t$ at different noise levels $t$. Then during sampling, we can compute $nabla_x log p(c|x)$ via backpropagation through the classifier.

In detail, if we assume the classifier outputs logits $h_i (x)$ for each class $i$, then the class probability is given by softmax:

$
  p_i (x) = "softmax"(h_i (x)) := exp(h_i (x)) / (sum_j exp(h_j (x))).
$

Then we can compute the gradient of log probability as

$
  nabla_x log p(c|x) = nabla_x h_c (x) - nabla_x log (sum_j exp(h_j (x))).
$

Or we can simply use $nabla_x h_c (x)$ as an approximation.

Furthermore, we can employ $nabla_x log p(x|c) = nabla_x log p(x) + s nabla_x log p(c|x)$, where $s > 1$ is a scaling factor to control the strength of the condition.

== Classifier Free Guidance (CFG)

The Classifier Guidance requires training a separate classifier model, which is inconvenient. Instead, we can train a single conditional diffusion model that takes condition $c$ as input, and an unconditional model by randomly dropping the condition during training. This is called Classifier Free Guidance (CFG).

Define a conditional denoiser

$
  D_theta (x_t, sigma, c) -> hat(x_0)
$
that takes condition $c$ as input. The denoiser is able to predict both conditional and unconditional outputs by setting $c$ to $emptyset$, a special token denoting no condition.

By Tweedie's formula, we can derive the score function from the denoiser as
conditional score

$
  nabla_x log p_theta (x_t, c) = (D_theta (x_t, sigma, c) - x_t) / sigma^2
$

and unconditional score

$
  nabla_x log p_theta (x_t) = (D_theta (x_t, sigma, emptyset) - x_t) / sigma^2.
$

Finally, to obtain the conditional score function, we combine the two scores as

$
  nabla_x log p(x|c)
  =& nabla_x log p_theta (x_t) + S(nabla_x log p_theta (x_t, c) - nabla_x log p_theta (x_t)) \
   =& 1/sigma^2 (S D_theta (x_t, sigma, c) + (1 - S) D_theta (x_t, sigma, emptyset) - x_t).
$

where $S >= 1$ is a scaling factor to guide the strength of the condition.
