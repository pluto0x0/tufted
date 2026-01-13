#import "../index.typ": template, tufted
#show: template.with(title: "Diffusion")

#let NN = $cal(N)$
#let vm(x) = $bold(upright(#x))$
#let xx = $vm(x)$
#let dt(x) = $accent(#x, dot)$
#let LL = $cal(L)$
// #let dp(x, y) = $chevron.l #x, #y chevron.r$
#let dp(x, y) = $#x^top#y$
#let lint = $limits(integral)$

#set math.equation(numbering: "1.")

= ODE and SDE

For an Ordinary Differential Equation (ODE)

$
  (dif x(t))/(dif t)=f(x(t), t),
$

the discrete time approximation with $i approx t/(Delta t)$ is given by

$
  x_i - x_(i-1) = f(x_(i-1), i-1)Delta t.
$

And for a Stochastic Differential Equation (SDE)

$
  (dif xx(t))/(dif t)=f(t,xx) + g(t,xx)xi(t)
$

where $f$ is drift, $g$ is diffusion term and $xi(t)$ is a random process such as a Weiner process: $W_(t+u)-W_t~NN(0, u)$, $W_0=0$).

This SDE gives us

$
  dif xx = f(t,xx) dif t + g(t, xx) dif vm(w)
$<eq:sde>

where $dif vm(w) = xi(t) dif t$.
In the SDE, $xx$ can not be solved simply by integration because $dif vm(w)$ is random.

== Reverse SDE

To solve the SDE, we want to find the distribution $p_t (xx)$ at each time $t$.

*Andersen's theorem*
The reverse SDE of @eq:sde is

$
  dif xx = [f(xx, t) - g(t)^2 nabla_xx log p_t (xx)] dif t + g(t) overline(vm(w))
$

where $overline(vm(w)) = vm(w) _(t-1)-vm(w)_t$ is the reverse Weiner process.
A reverse SDE means that it has the same distribution $p_x (xx)$ at each time $t$ starting from $xx_T ~ p_T (xx)$ and ending at $xx_0 ~ p_0 (xx)$.

Intuitively, the evolution of probability distribution of a SDE is a guided diffusion process, thus the reverse SDE will require a guidance term $ nabla_xx log p_t (xx)$ to shrink the distribution back.
#footnote[
  This is similar to the Langevin dynamics mentioned in the score maching diffusion model, i.e.
  with knowing $xx_T$ and $nabla_xx log p_t (xx)$, we can solve the reverse SDE to get $p_t (xx)$.
]

== Numerical Methods

For ODEs, we solve $x$ numerically by *Eular's method* (first order) 
$
  x_(n+1)=x_n+Delta t f(x_n,n),
$

*Runge–Kutta method* (second order)

$
  f_1=f(x_(i-1), t_(i-1)), f_2=f(x_(i-1)+alpha/2 f_1, t_(i-1)+(Delta t)/2).
$

Or higher order methods like 4th order Runge-Kutta (RK4) method.

For SDEs, we have the *Euler-Maruyama method* similar to Eular's method:

$
  xx_(n+1)=xx_n + f(t_n, xx_n) Delta t + g(t_n) Delta w_n.
$

It's easy to see that the forward process of diffusion models

$
  x_i = sqrt(1-beta_i) x_(i-1) + sqrt(beta_i) z_i
$

correspondes to a SDE:

$
  dif xx = (-beta(t))/2 xx dif t + sqrt(beta(t)) dif vm(w)
$

with approximation $sqrt(1-beta_i)-1 approx -1/2 beta(t_i)x_(i-1)$ for small $beta$.
And the reverse SDE corresponding to the backward process is

$
  d xx = beta(t) [xx/2 - nabla_xx log p_t (xx)] dif t + sqrt(beta(t)) abs(overline(vm(w))).
$

Finally, the discrete sampling process correspondes to the Euler-Maruyama method of the reverse SDE:

$
  xx_i
  = (1+beta_i/2)[xx_i + beta_i/2 nabla_x log p_i (xx_i)]+sqrt(beta_i) vm(z)_i \
  approx 1/sqrt(1-beta_i)[xx_i + beta_i/2 nabla_x log p_i (xx_i)]+sqrt(beta_i) vm(z)_i
$

= Flow Matching

The goal of Flow Matching is to design a *vector field* (or the equivalent ODE) that transforms a sample-able inital distribution (e.g., Gaussian) to the target data distribution.

First we define the following terms:

+ State: $xx(t) = xx_t$ where $t in [0, 1]$.
+ Flow: $Psi_t(dot)$, the location of a initial state at time $t$.
+ Vector field and ODE: $(dif xx_t)/(dif t) = u_t (xx_t)$, where $u_t (xx_t)$ is the vector field.
+ Trajectories: $dif/(dif t) Psi(xx_0) = u_t (xx_t)$ with $xx_0 ~ p_0$.

== Conditional Flow Matching

Assume we sample a data $z_j$ as a condition of the flow, i.e. we want the trajectory to end at $z_j$ at $t=1$ starting from $xx_0 ~ p_0$ at $t=0$.
Thus we can construct a simple linear trajectory

$
  Psi(xx_0) = alpha_t z_j + beta_t xx_0
$

where $alpha_0 = 0, alpha_1 = 1, beta_0 = 1, beta_1 = 0$ are the interpolation coefficients.
And because $xx_0 ~ NN(0, I)$, we have $p_t (xx|z) ~ NN(alpha_t z, beta_t^2 I)$.

Since we have defined the trajectory, the next question is how to design a vector field $u_t (xx|z)$ that satisfies

$
  dif/(dif t) Psi(xx_0|z) = u_t (xx_t|z).
$

i.e., the vector field that pushes $xx_0 ~ p_0$ to $xx_1 ~ p_"data"$.

Let $y =alpha_t z + beta_t xx$, then

$
  u_t (y|z) = dif/(dif t) y = dt(alpha_t) z + dt(beta_t) xx,
$

and since $xx = 1/beta_t (y - alpha_t z)$, we have

$
  u_t (y|t) = dt(alpha_t) z + dt(beta_t)/beta_t (y - alpha_t z) = (dt(alpha_t) - dt(beta_t)/beta_t alpha_t)z + dt(beta_t)/beta_t y
$

== Continuity Equation

Continuity Equation describes the time evolution of the probability density function of a particle system flowing according to a vector field.
If $(dif xx_t)/(dif t) = u_t (xx_t)$, then the distribution $p_t (xx)$ satisfies 

$
  partial_t p_t (xx) = -nabla dot [p_t (xx) u_t (xx)]
$

where $nabla$ is the divergence operator defined as
$nabla dot f(xx) = (partial f_1)/(partial x_1) + (partial f_2)/(partial x_2) + dots + (partial f_n)/(partial x_n)$, which described the net "intensity" of the field (probability mass) flowing out of a point.

#tufed.margin-note[
  #image("img/div.png")
]

#tufed.margin-note[
  (#link("https://www.khanacademy.org/math/multivariable-calculus/multivariable-derivatives/divergence-and-curl-articles/a/divergence")[Image Credit])
]

#tufed.margin-note[
  Take $x$ direction as an example, if $(partial u) / (partial x) > 0$, it means the strength of probability mass flowing out of the point on the right side is larger than the strength flowing in from the left side, thus the net effect is probability mass is flowing out of the point, i.e., divergence is positive.
]

== Marginalize $u_t (xx|z) -> u_t (xx)$

Now we have already obtained the conditional flow $u_t (xx|z)$ that can transforms $xx_0 ~ p_0$ to $xx_1 = z$ given condition $z ~ p_"data"$.
Then we can move on to find a marginalized $u_t (xx)$ that transforms $xx_0 ~ p_0$ to $xx_1 ~ p_"data"$.

By continuity equation, we have the

$
  partial_t p_t (xx|z) & = - nabla dot.op [p_t (xx|z) u_t (xx|z)] \
  partial_t p_t (xx) & = - nabla dot.op [p_t (xx) u_t (xx)]
$

respectively for conditional and unconditional flows.
By substituting into $p_t (xx) = integral p_t (xx|z) p (z) dif z$ we have

$
  partial_t p_t (xx) = & partial_t integral p_t (xx|z) p (z) dif z\
  = & integral [partial_t p_t (xx|z)] p (z) dif z\
  = & integral - nabla dot.op [p_t (xx|z) u_t (xx|z)] p (z) dif z\
  = & - nabla dot.op [integral p_t (xx|z) u_t (xx|z) p (z) dif z]
$

By eliminating the divergence operator, we have

$
  & integral p_t (xx|z) u_t (xx|z) p (z) dif z = p_t (xx) u_t (xx)\
  => & u_t (xx) = frac(integral p_t (xx|z) u_t (xx|z) p (z) dif z, p_t (xx)) = bb(E)_(z ~ p_t (z|x)) [u_t (xx|z)]
$

which finnaly gives us the marginalized vector field.

== Loss Function

We use the a flow prediction model $u_t^theta$ to approximate $u_t$. We can define the Flow Matching loss as follows, which is the MSE between the predicted vector field and the true vector field.

$
  LL_"FM" (theta) = EE_(t~"U"[0,1], xx~p_t (xx))[norm(u_t^theta (xx) - u_t (xx))^2].
$

To sample from $p_t (xx)$, we can employ the ancestral sampling trick by first sampling $z ~ p_"data"$, and then $xx ~ p_t (xx|z)$.

=== Conditional Loss

In the Flow Matching loss, we need to explicitly sample a data point $z$ in order to sample $xx ~ p_t (xx)$, thus we can instead define a conditional Flow Matching loss as follows:

$
  LL_"CFM" (theta) = EE_(t~"U"[0,1], z~p_"data", xx~p_t (xx|z))[norm(u_t^theta (xx) - u_t (xx|z))^2]
$

A beautiful property of the two losses is that they have the same gradient with respect to $theta$, i.e. $LL_"FM" - LL_"CFM"$ is a constant independent of $theta$. We can prove this by firstly expanding the $norm(dot)^2$ term:

$
  norm(u_t^theta (xx) - u_t (xx|z))^2 = norm(u_t^theta (xx))^2 + norm(u_t (xx|z))^2 - 2 dp(u_t^theta (xx), u_t (xx|z))
$

where $norm(u_t^theta (xx))^2$ is the same for both losses,
and  $norm(u_t (xx|z))^2$ is independent of $theta$. Thus for both losses, we only care about the 3rd term:

$
  EE_(t, z, x) [u_t^theta (x)^top u_t (x|z)]
  = & integral_0^1 integral integral u_t^theta (x)^top u_t (x|z) p_t (x|z) dif x p (z) dif z dif t med\
  = & integral_0^1 integral integral u_t^theta (x)^top u_t (x|z) p_t (x|z) p (z) dif z dif x dif t med\
  = & integral_0^1 integral u_t^theta (x)^top integral u_t (x|z) p_t (x|z) p (z) dif z dif x dif t med\
  = & integral_0^1 integral u_t^theta (x)^top u_t (x) p (x) dif x dif t med\
  = & EE_(t, x) [u_t^theta (x)^top u_t (x)],
$

which is exactly the 3rd term in $LL_"FM"$.
Thus, we have $LL_"FM" = LL_"CFM" + C$ where $C$ is a constant independent of $theta$, and thus they have the same gradient with respect to $theta$.

== Algorithm

We can easily write down the training algorithm based on the conditional Flow Matching loss:

+ Sample $t ~ U[0,1]$, $z ~ p_"data"$, $epsilon ~ NN(0, I)$ and $xx <- t z + (1-t) epsilon$ \
+ GD step on $nabla_theta norm(u_t^theta (xx) - (z-epsilon))^2$.

= Fokker-Planck Equation

$
  (partial p) / (partial t) = -nabla dot (f p) + 1/2 nabla^2 (g g^top p)
$

where $f$ is drift, $g$ is diffusion term, $nabla dot$ is divergence and $nabla^2$ is Laplacian operator, the divergence of the gradient. The eqation describes the time evolution of the probability density function of the random variable xx governed by the SDE. Same concept with continuity equation.
Fokker-Planck Equation tells us a ODE can also describe the probability density evolution of a SDE.

For a original ODE $(dif xx_t)/(dif t) = u_t (xx_t)$ with $xx_0 ~ p_0$.
The equivalent SDE is $dif xx_t = [u_t (xx_t) - sigma_t^2/2 nabla_xx log p_t (xx_t)] dif t + sigma(t) dif vm(w)$.
