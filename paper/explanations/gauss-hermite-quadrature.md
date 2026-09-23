# How Gauss–Hermite Quadrature Approximates an Integral

`np.polynomial.hermite.hermgauss(n)` computes the sample points and weights for
Gauss–Hermite quadrature. These correctly integrate polynomials of degree
`2n − 1` or less over `(−∞, +∞)` against the weight function `w(x) = exp(−x²)`.

This note explains *how* that constitutes an approximation of an integral,
built up from the ground rather than asserted.

---

## The thing we're really trying to do

Forget the weight function for a second. The oldest idea for approximating any
integral is: sample the function at some points, multiply each value by how much
"width" that point represents, and add up. That's a Riemann sum:

$$\int f(x)\,dx \approx \sum_i f(x_i)\,\Delta x_i.$$

Every quadrature rule is a smarter version of that same move: **pick points
$x_i$, pick weights $w_i$, and approximate the integral by $\sum_i w_i f(x_i)$.**
The entire game is choosing the points and weights well.

A Riemann sum chooses badly on purpose — evenly spaced points, equal widths —
because it's simple. You need a lot of points before it's accurate. Gaussian
quadrature asks the opposite question: if I'm only *allowed* $n$ points, where
should I put them and what weights should I give them to be as accurate as
possible?

---

## What "as accurate as possible" means — and how you'd pin the numbers down

Here is the concrete recipe, and seeing it is what makes the whole thing click.

Suppose you allow yourself just $n = 2$ points to approximate
$\int f(x)\,e^{-x^2}\,dx$. You have four unknowns: two node positions
$x_1, x_2$ and two weights $w_1, w_2$. Your approximation is

$$\int f(x)\,e^{-x^2}\,dx \approx w_1 f(x_1) + w_2 f(x_2).$$

Now impose a demand: **make this exact for as many simple functions as
possible.** The simplest functions are $1, x, x^2, x^3$. Four unknowns, so you
can demand exactness for four functions. Each demand is one equation, where the
left side is a genuine integral you can compute by hand:

| $f(x)$ | equation | value of the integral |
|--------|----------|-----------------------|
| $1$    | $w_1 + w_2$ | $\int e^{-x^2}\,dx = \sqrt{\pi}$ |
| $x$    | $w_1 x_1 + w_2 x_2$ | $\int x\,e^{-x^2}\,dx = 0$ |
| $x^2$  | $w_1 x_1^2 + w_2 x_2^2$ | $\int x^2 e^{-x^2}\,dx = \sqrt{\pi}/2$ |
| $x^3$  | $w_1 x_1^3 + w_2 x_2^3$ | $\int x^3 e^{-x^2}\,dx = 0$ |

Four equations, four unknowns. Solve the system and out come specific numbers:

$$x_1 = -\tfrac{1}{\sqrt 2}, \quad x_2 = +\tfrac{1}{\sqrt 2}, \quad
w_1 = w_2 = \tfrac{\sqrt \pi}{2}.$$

**Those are precisely the numbers `hermgauss(2)` returns.** That's all the
function is doing — it has solved this system for you, for whatever $n$ you ask,
and handed back the resulting nodes and weights.

> The `hermgauss` implementation doesn't literally solve these equations (it uses
> the fact that the optimal nodes are the roots of the $n$-th Hermite polynomial,
> a cleaner route to the same answer), but the *meaning* of the numbers it returns
> is exactly the solution to that system.

---

## Why it then works on functions that *aren't* polynomials

The construction only forced exactness on $1, x, x^2, x^3$. But you want to
integrate some arbitrary smooth $g$, not a cubic. Why should a rule tuned on
polynomials do anything sensible for $g$?

Because a smooth function *behaves locally like a polynomial* — that's what a
Taylor expansion says. Write $g$ as its low-degree polynomial part plus a
remainder. The rule integrates the polynomial part with *zero error* (that's
what we built it to do). So the only error the rule makes is on the remainder —
the part of $g$ that a cubic can't capture. If $g$ is smooth, that remainder is
small, so the error is small.

With $n = 2$ the rule nails everything up to degree $3$ and only errs on
degree-4-and-up wiggliness; push to $n = 15$ and it nails everything up to
degree $29$, leaving an even tinier remainder. That's the sense in which
"exact for polynomials up to $2n-1$" is the whole quality guarantee: **the
higher that degree, the more of your function is absorbed error-free, and the
less is left to approximate.**

> This is also why a **kink** is the villain. A function like
> $\max(1 - F,\,0)$ is *not* well-approximated by any polynomial near its corner
> — a polynomial can't have a sharp kink — so the "remainder is small" argument
> weakens right there. That is the one place the polynomial-exactness guarantee
> frays, and the reason a cross-check (grid-doubling, Monte-Carlo comparison) is
> needed around it.

---

## Where the weight function comes in

Notice every integral in the table had the $e^{-x^2}$ baked in — it was
$\int x^2 e^{-x^2}\,dx$, not $\int x^2\,dx$. **The weight function is absorbed
into the weights $w_i$.** You never evaluate $e^{-x^2}$ when you use the rule;
you just compute $\sum_i w_i f(x_i)$, and the $w_i$ already "know" about the
Gaussian weighting because they were solved against integrals that contained it.

That's what makes it tailor-made for **expectations**. An expectation *is* an
integral-against-a-density,

$$\mathbb{E}[g(Z)] = \int g(z)\cdot(\text{density})\,dz,$$

and the Hermite weight $e^{-x^2}$ is — after a change of variables — that
density. So the rule is purpose-built to turn "average of $g$ over a normal
shock" into a short weighted sum.

---

## The change of variables (the one practical subtlety)

The Hermite weight is $e^{-x^2}$, but the standard-normal density is
$(2\pi)^{-1/2} e^{-z^2/2}$. These don't coincide — the exponent differs by a
factor of two, plus a normalising constant. Setting $z = \sqrt 2\,x$ reconciles
them:

$$\mathbb{E}[g(Z)] = \frac{1}{\sqrt\pi}\int g(\sqrt 2\,x)\,e^{-x^2}\,dx
\approx \sum_i \frac{w_i}{\sqrt\pi}\,g(\sqrt 2\,x_i), \qquad Z \sim \mathcal N(0,1).$$

Two things to read off:

- The nodes you actually evaluate $g$ at are $\sqrt 2\,x_i$, **not** $x_i$.
- The probability weights are $w_i / \sqrt\pi$, which **sum to exactly one**
  (the raw Hermite weights sum to $\sqrt\pi$). After this normalisation they are
  a genuine probability distribution over the nodes — an honest weighted average,
  not arbitrary coefficients.

> NumPy also ships the *probabilists'* convention as
> `np.polynomial.hermite_e.hermegauss`, whose weight is $e^{-x^2/2}$ — matching
> the normal's shape directly, so the nodes need no $\sqrt 2$ rescaling and the
> weights only a $/\sqrt{2\pi}$. Same rule, different bookkeeping.

---

## The one-line summary

`hermgauss(n)` hands you $n$ points and $n$ weights chosen so that
$\sum_i w_i f(x_i)$ reproduces $\int f(x)\,e^{-x^2}\,dx$ **exactly** whenever
$f$ is a polynomial of degree $< 2n$, and **approximately** — with an error
controlled by how far $f$ departs from a polynomial — for everything else.
Approximating the integral is nothing more than evaluating that weighted sum.
