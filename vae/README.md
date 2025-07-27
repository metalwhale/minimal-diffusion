# Variational Autoencoder
Generative models are defined by their ability to learn the distribution $p(x)$ of a given training dataset, and VAE (variational autoencoder) is one example of such a model. But how does a VAE learn this? Let's find out together. The main purpose of this article is to explore DDPM, but we'll start with VAE first, since they're simpler, and then move on to DDPM later.

You may ask: why bother writing another article about VAE when there are already tons of them out there? The reason is that most of those blogs or papers immediately dive into math-heavy explanations, flooding readers with equations about ELBO, KL divergence, Gaussians, and more - which can be difficult for newcomers to follow.

I truly respect those authors and thank them for sharing their knowledge with us. However, I've never felt fully satisfied when reading such articles. I don't just want to comprehend the math, I want to see a bigger picture. I don't just want to grasp what the authors think, I also want to figure out how they came up with their solutions. Understanding what others have done makes us followers, but understanding the general principles of solving a problem can someday make us leaders.

Let's define the problem in clear and simple language, and then gradually work toward explaining how a VAE can learn $p(x)$, using intuitive reasoning and as little math as possible. While breaking my explanation into multiple steps, I will try to connect all crucial terms, because connecting everything together is key to achieving a general view. If you just want to see the short version, you can jump straight to section ["2. Conclusion"](#2-conclusion), but if you are curious like me and want to gain a deeper understanding, come with me.

<details><summary>Prerequisites</summary>

I assume the readers have some basic knowledge of [vanilla autoencoder](https://en.wikipedia.org/wiki/Autoencoder) and [VAE](https://en.wikipedia.org/wiki/Variational_autoencoder) as it is difficult to explain everything from scratch. For those who are not familiar with them, this article may be a bit hard to follow, but you are still very welcome.
</details>

## 1. Understanding VAE Learning
### 1-1. Learning the Target Distribution
Suppose we have a *target distribution* $p(x)$, which is unknown. We will never have direct access to its form or equation, but we can sample from it. We also have a model $p_\theta(x)$, which could be a neural network or any other parameterized function with parameter $\theta$.

> [!TIP]
> The goal of learning $p(x)$ is to adjust $\theta$ during training so that $p_\theta(x_i)$ closely approximates $p(x_i)$ for all observed sample $x_i$.

<details><summary>Caveat</summary>

Remember that this tip is true during training, but the ultimate goal of a generative model is usually to approximate the overall distribution $p(x)$, rather than matching probabilities of individual observed data points.
</details>

Whenever someone says: "we are training a generative model", they actually mean: "we are training a model that can approximate the target distribution", or more precisely, a model that can produce samples whose distribution is close to that of observed samples from the target distribution.

### 1-2. Maximum Likelihood Estimation (MLE)
What can we do to enable a model to learn $p(x)$? Generally speaking, we achieve this through [*maximum likelihood estimation (MLE)*](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).

Likelihood function is defined as the probability of observing a data point $x_i$ given a parameter $\theta$. Training a model with MLE means finding the optimal $\hat{\theta}$ so that the model assigns the highest possible value to the joint likelihood $\prod_{i}p_\theta(x_i)$ across all observed $x_i$.

$$
\hat{\theta}=\arg\max_{\theta}\prod_{i}p_\theta(x_i)
$$

In practice, it is encouraged to rewrite this formula using log-likelihood function because it is easier to compute.

$$
\hat{\theta}=\arg\max_{\theta}\log\left(\prod_{i}p_\theta(x_i)\right)=\arg\max_{\theta}\sum_{i}\log p_\theta(x_i)
$$

But why do we want to maximize the log-likelihood? How does it help us learn $p(x)$? Simply put: suppose we have observed several samples from $p(x)$. Maximizing the log-likelihood is like saying: "Here are samples we observed. There may be many other samples we haven't seen yet, or the true distribution might differ from what we observed. However, our best guess for a distribution close to $p(x)$ is to assume that what we observe has the highest probability". If something happens many times, it must have a high probability, so we simply maximize that.

Still not convinced yet? Let's take a simple example. Suppose we have a coin that follows a target distribution $p(x)$, where $x$ can take only two values: $H$ for heads and $T$ for tails. After tossing the coin 6 times, we observe the following results: $H-T-H-T-H-H$. So, our problem of learning $p(x)$ boils down to optimizing a parameter $\theta$ using MLE:

$$
\begin{align}
\hat{\theta}&=\arg\max_{\theta}\left(\log p_\theta(H)+\log p_\theta(T)+\log p_\theta(H)+\log p_\theta(T)+\log p_\theta(H)+\log p_\theta(H)\right) \\
&=\arg\max_{\theta}\left(4\log p_\theta(H)+2\log p_\theta(T)\right)
\end{align}
$$

We only know two things: the probabilities of all event $p_\theta(H)$ and $p_\theta(T)$ must sum to 1 (which is obvious since $p_\theta$ is a probability function), and the observations we have above. In this case, the coin seems unfair: we got 4 heads and 2 tails, the probabilities are estimated as $p_\theta(H)=\frac{4}{6}=\frac{2}{3}$ and $p_\theta(T)=\frac{2}{6}=\frac{1}{3}$ (see ["Appendix 3-1."](#3-1-solution-to-the-coin-toss-example)). Is this correct? What if the target distribution is totally different, e.g., $p(H)=\frac{1}{10}$ and $p(T)=\frac{9}{10}$? Unfortunately, we will never know for sure. But we can improve the accuracy of our model by doing a (hopefully) very simple thing: collecting more data. This is the essence of [Law of large numbers](https://en.wikipedia.org/wiki/Law_of_large_numbers): the more samples we observe from the target distribution, the better chance we have of learning it.

> [!TIP]
> In a generative model, $p_\theta(x)$ is trained with MLE by maximizing $\log p_\theta(x_i)$ for every $x_i$.

You may have noticed in many articles about generative models, including this one, that MLE is often expressed not in its full form over all observed samples but only for a single example (i.e., without $\sum_{i}$). This is perfectly fine: in practice, we compute the individual log-likelihoods and then *average* them over all observed data, which makes the objective independent of the dataset size. We still maximize the sum over all samples, not just a single one. By doing this, values that appear more frequently (i.e., the same event repeated in different observations $x_i$) gradually dominate the distribution. For example, in the coin toss case above, we perform MLE over all $x_i$, but since $H$ happens many times, the model will prioritize $H$ over $T$, leading to $p_\theta(H)$ eventually becoming larger than $p_\theta(T)$.

### 1-3. Latent Variable
Ok, then how do we train $p_\theta(x)$ with MLE? By introducing a *latent variable* $z$. The detailed reason for leveraging $z$ when training a VAE is beyond this article's scope, but we can understand that this variable serves as a hidden, compressed representation of the data. Most importantly, it should be *tractable* so that we can control it.

<details><summary>Considerations</summary>

One may ask: why not just learn $p_\theta(x)$ directly using MLE? This is a subtle question, and honestly, I haven't found a fully convincing answer. My guess is that learning that way only gives us a density function without telling us how to sample from it. It is still theoretically possible to sample from any density model, but in most cases, it is neither straightforward nor computationally efficient. Another challenge is that in many real-world problems, $x$ is high-dimensional, which makes modeling $p_\theta(x)$ directly very difficult. Perhaps this is why the latent variable $z$ is intentionally introduced in VAE, to make both training and generation more feasible.
</details>

$$
p_\theta(x)=\int p_\theta(x,z)dz=\int p_\theta(x\mid z)p(z)dz
$$

What we are doing here is *marginalizing out* the latent variable $z$, and this is also how we generate samples from $p_\theta(x)$. At first, this integral may look scary, but it isn't. Let's break it down into smaller terms:
- $p(z)$: This represents drawing a sample $z$ that follows the probability distribution of the latent variable, which is also called the *prior distribution* $p(z)$. It can be chosen as a simple distribution (such as a Gaussian) and does not need to be learned (hence, it does not include any parameters), so sampling it is straightforward.
- $p_\theta(x\mid z)$: After obtaining a sample $z$, we treat it as a fixed value and draw a sample $x$ from $p_\theta(x\mid z)$. This is a conditional distribution that provides the probability of each value of $x$ given a fixed input $z$, and needs to be learned during training.

> [!TIP]
> We learn $p_\theta(x)$ indirectly by learning a conditional likelihood $p_\theta(x\mid z)$, which is called the *decoder*.

But sampling $z$ from $p(z)$ and then $x$ from $p_\theta(x\mid z)$ defines the joint distribution $p_\theta(x\mid z)p(z)=p_\theta(x,z)$, whereas what we actually need is the marginal distribution $p_\theta(x)$. So, what are we missing here? This is where the integral $\int...dz$ comes in: it considers all possible latent values $z$, weights each by how likely it is under the prior $p(z)$, and marginalizes out $z$ to obtain the likelihood of $x$.

*Marginalizing out* every $z$ is a formal probability operation, not something we do directly at generation time. During generation, we only sample individual $x$, which won't statistically represent how well $p_\theta(x)$ matches $p(x)$. But that's ok, since we are dealing with distributions, a single or a few samples don't tell us whether $p_\theta(x)$ has truly learned $p(x)$. What really matters is that, as the number of samples grows, the learned distribution $p_\theta(x)$ should approach $p(x)$. The integral expresses this "averaging over all $z$" exactly.

### 1-4. Evidence lower bound (ELBO)
So far, we know that in order to learn $p(x)$, we need to apply MLE to the generative model $p_\theta(x)=\int p_\theta(x\mid z)p(z)dz$. In other words, we want to find the optimal $\hat{\theta}$ that maximizes this integral. Our problem is that it is not easy to evaluate, or as the authors of VAE put it, *intractable*. We therefore need to find another way to express $p_\theta(x)$ in tractable terms, while also ensuring it naturally involves $p_\theta(x\mid z)$ and $p(z)$, since they are part of the model definition and are required when generating samples from $p_\theta(x)$ during decoding.

The solution the authors of VAE arrived at is to use [*evidence lower bound (ELBO)*](https://en.wikipedia.org/wiki/Evidence_lower_bound).
<details><summary>Disclaimer</summary>

I'm trying to connect ELBO with the terms required during decoding. In many VAE articles, authors simply show ELBO and the decoding process without explaining why both include $p_\theta(x\mid z)$ and $p(z)$. This can't be a coincidence, they both involve these terms because they are part of the underlying generative model $p_\theta(x,z)$. I assume there may be alternative ways to write ELBO or other methods, but this standard form follows directly from the model, so a natural structural connection to decoding terms emerges.
</details>

```math
\log p_\theta(x)\ge\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]-D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p(z)\right)
```

Great! Our new expression includes both $p_\theta(x\mid z)$ and $p(z)$, as expected. But we also see a new term introduced: $q_\phi(z\mid x)$, which we will refer to as the *encoder* (with a parameter $\phi$) from now on. The reason we introduce $q_\phi(z\mid x)$ is to approximate the true posterior $p(z\mid x)$ (see ["Appendix 3-2."](#3-2-detailed-derivation-of-elbo)), which is intractable. What is happening here? Let's break things down and take a closer look. They are so elegant and all make sense:
- The LHS is our model expressed in terms of log-likelihood. As discussed above, we want to find a way to maximize this term for every sample $x_i$.
- The RHS is ELBO. A detailed derivation of ELBO is beyond this article's scope, but all you need to remember is: ELBO stands for "Evidence Lower Bound", meaning that by maximizing ELBO, we indirectly maximize the LHS and achieve our goal. ELBO consists of two different terms with distinct roles, but they are all connected to each other:
    - *Reconstruction term* $`\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]`$

        This term measures the likelihood of the decoder $p_\theta(x\mid z)$, given a latent variable $z$ sampled from the encoder $q_\phi(z\mid x)$. Please keep in mind: during training, the decoder $p_\theta(x\mid z)$ takes input $z$ sampled from the encoder $q_\phi(z\mid x)$, not from the prior $p(z)$ as at generation time.

        Maximizing this term contributes to the maximization of the ELBO, as it ensures that the learned decoder is able to generate data with high likelihood.
    - *Prior matching term* $`D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p(z)\right)`$

        This term measures how closely the encoder $q_\phi(z\mid x)$ matches the prior $p(z)$.

        Minimizing this term (note the minus sign $-$) also contributes to the maximization of the ELBO, as it encourages the learned encoder to become similar to the prior. This makes sense because it allows us, during decoding at generation time, to replace the encoder with the prior as a source distribution for input $z$ to the decoder, which aligns with how we defined the decoding process above.

        ($D_{\mathrm{KL}}$ is [*KL divergence*](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence), which has a low value when its two input distributions are similar and a high value when they differ significantly).

> [!TIP]
> VAE doesn't apply MLE directly to $p_\theta(x)$, but indirectly by maximizing ELBO, which involves two terms: reconstruction term for training the decoder $p_\theta(x\mid z)$, and prior matching term for training the encoder $q_\phi(z\mid x)$.

#### 1-4-1. More About the Reconstruction Term
The reconstruction term $`\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]`$ illustrates the generation process: we feed an input $x$ into the encoder $q_\phi(z\mid x)$ to sample a latent variable $z$, and then pass $z$ into the decoder $p_\theta(x\mid z)$ to produce an output $\tilde{x}$. While this describes a single generation sample, during training the expectation $\mathbb{E}$ captures overall reconstruction ability across all latent variables sampled from the encoder distribution.

In many articles about VAE, the reconstruction term is often described as "ensuring that $\tilde{x}$ is as close as possible to $x$". While this explanation sounds reasonable and intuitive, it is not immediately obvious. The reconstruction term is actually a log-likelihood, which tells us only that "we must generate outputs that are as probable as possible", but it does not explicitly require us to "generate outputs that are identical to the corresponding inputs". "Reconstruction" is a word borrowed from vanilla autoencoder, where the goal is to match inputs and outputs literally. However, in VAE, how does this "reconstruction" relate to the log-likelihood? Let's work this out with an example.

You can read more about this in [Chan et al. (2024), "1.3 Optimization in VAE"](#4-3-tutorial-on-diffusion-models-for-imaging-and-vision), but I will simplify it here. Suppose we choose the decoder to be a [Gaussian distribution](https://en.wikipedia.org/wiki/Normal_distribution): $p_\theta(x\mid z)=\mathcal{N}(x;\mu,\sigma^2)=\mathcal{N}(x;f_\theta(z),\sigma^2)$, where $f_\theta(z)$ is a decoder network and $\sigma^2$ is a fixed variance. We can interpret this as follows: $p_\theta(x\mid z)$ is a function that takes $z$ as input and generates $x$ as output, so that $x$ "depends" on the "condition" $z$. Since a Gaussian distribution has two parameters - a mean $\mu$ and a variance $\sigma^2$, we need to find a way to map $z$ to these parameters. Here, we fix the variance, so the decoder only needs to map $z$ to the mean, which is achieved by $f_\theta(z)$.

Recall that our goal is to maximize the reconstruction term, that is, to maximize $\log p_\theta(x\mid z)$ for each $x$ and $z$.

```math
\begin{align}
\log p_\theta(x\mid z)&=\log \mathcal{N}(x;f_\theta(z),\sigma^2) \\
&=\log\left(\frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{(x-f_\theta(z))^2}{2\sigma^2}}\right) \\
&=-\frac{(x-f_\theta(z))^2}{2\sigma^2}-\log\left(\sqrt{2\pi\sigma^2}\right)
\end{align}
```

If we define the decoder's output as $\tilde{x}=f_\theta(z)$, then maximizing the reconstruction term is equivalent to minimizing the mean squared error $(x-\tilde{x})^2$, or "reconstructing" $x$ by generating an output $\tilde{x}$ that is as close as possible to $x$. A similar conclusion holds if we instead use a [Bernoulli distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution) as the decoder (see ["Appendix 3-3."](#3-3-use-a-bernoulli-distribution-as-the-decoder))

> [!TIP]
> In a VAE, encouraging outputs to match inputs is not arbitrary. It naturally arises from the reconstruction term $`\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]`$, which maximizes the decoder's log-likelihood.

<details><summary>Considerations</summary>

I found an interesting discussion [here](https://stats.stackexchange.com/questions/368001/is-the-output-of-a-variational-autoencoder-meant-to-be-a-distribution-that-can-b). Their question not only clarifies the meaning of "reconstruction" but also asks whether there are other ways to design the decoder probabilistically, beyond simply reconstructing the input.
</details>

#### 1-4-2. More About the Prior Matching Term
The prior matching term $`D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p(z)\right)`$ encourages the encoder to approximate the prior, but this does not mean we want $q_\phi(z\mid x)$ to equal $p(z)$ for every $x$. Instead, we want the "aggregated" posterior $\int q_\phi(z\mid x)p_\theta(x)dx$ to match $p(z)$. In other words, for each $x$, its corresponding posterior $q_\phi(z\mid x)$ can still be different, encoding meaningful information about that specific $x$, but across the entire dataset, distribution of latent samples should resemble $p(z)$.

Here lies something very interesting, though not immediately obvious: what happens if every posterior is identical for all $x$? Is that possible? It turns out this question plays a critical role in how we understand VAE. In many articles, ELBO is presented as if it simply "appears naturally", something we are expected to accept without question. It feels as though it pops out of thin air and tells us: "Hey, I'm ELBO, all you need to do is memorize me, don't wonder why I look this way". Personally, I don't accept that. There must be a reason for this form, and we cannot truly claim to understand VAE without knowing why the ELBO is written this way.

This leads us to two questions:
- Is there another way to write the ELBO, especially by changing the prior matching term?
- What disadvantages would this alternative form have compared to the standard form?

Answering the first question: yes, there is another way! To see this clearly, please refer to the detailed derivation of the standard form in ["Appendix 3-2."](#3-2-detailed-derivation-of-elbo). But to simplify: mathematically, we can replace $q_\phi(z\mid x)$ with $q_\phi(z)$, yielding another form of ELBO:

```math
\log p_\theta(x)\ge\mathbb{E}_{q_\phi(z)}\left[\log p_\theta(x\mid z)\right]-D_{\mathrm{KL}}\left(q_\phi(z)\,\|\,p(z)\right)
```

In this form, the prior matching term $`D_{\mathrm{KL}}\left(q_\phi(z)\,\|\,p(z)\right)`$ no longer depends on $x$, it is identical across all inputs. This also applies to the reconstruction term $`\mathbb{E}_{q_\phi(z)}\left[\log p_\theta(x\mid z)\right]`$: during training, the decoder only receives latent variables sampled from a distribution $q_\phi(z)$ that does not depend on $x$. This exposes a problem: training VAE in such a way is ineffective, which answers the second question. It is equivalent to discarding the encoder and simply training the decoder on samples from the prior. As a result, the model can no longer capture differences between samples in the target distribution. The decoder becomes "lazy", collapsing to a mode where it generates almost similar outputs every time we sample.

As you can see, nothing in machine learning "just comes out of nowhere". There is always a reason behind each technique. Unfortunately, authors often do not put much effort into showing the entire process they went through, the decisions they made, and the considerations they weighed before arriving at the final results. Understanding all of this takes time, but it makes our journey into AI much more enjoyable and valuable.

> [!TIP]
> The prior matching term $`D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p(z)\right)`$ includes two crucial elements:
> - KL divergence, which pushes latent variables to align with the prior distribution.
> - Dependence on the input $x$, which ensures that each input maintains its own distinct latent representation instead of collapsing into a single shared distribution.

<details><summary>Disclaimer</summary>

Honestly, I could not find any article directly supporting these claims. Much of this is my hypothesis (with some help from LLMs), so please take it with a grain of salt.
</details>

## 2. Conclusion
Generally speaking, what we have done so far can be summarized step by step as follows:
1. Define the training objective

    Our goal is to train a generative model to approximate the target distribution $p(x)$ by maximizing the log-likelihood $\log p_\theta(x)$, where $\theta$ is the model parameter to optimize.

2. Introduce a latent variable

    We introduce a latent variable $z$ and express the marginal likelihood as $p_\theta(x)=\int p_\theta(x\mid z)p(z)dz$, where $p(z)$ is a prior distribution (usually simple and does not need to be learned).

3. Optimize via ELBO

    Directly maximizing $\log p_\theta(x)$ is intractable due to the integral over $z$. Instead, we maximize ELBO defined as: $`\log p_\theta(x)\ge\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]-D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p(z)\right)`$.

    This involves learning the encoder $q_\phi(z\mid x)$ to produce a simple latent distribution close to the prior $p(z)$, and the decoder $p_\theta(x\mid z)$ to reconstruct $x$ from $z$.

## 3. Appendix
### 3-1. Solution to the Coin Toss Example

We want to find $\theta$ that satisfies:
```math
\hat{\theta}=\arg\max_{\theta}\left(4\log p_\theta(H)+2\log p_\theta(T)\right)
```
Substituting $p_\theta(T)$ with $1-p_\theta(H)$, we obtain:
```math
\hat{\theta}=\arg\max_{\theta}\left(4\log p_\theta(H)+2\log\left(1-p_\theta(H)\right)\right)
```
Finding $\hat{\theta}$ is equivalent to finding $p_\theta(H)$ that maximizes this function:
```math
f\left(p_\theta(H)\right)=4\log p_\theta(H)+2\log\left(1-p_\theta(H)\right)
```
We can do this by taking the derivative and setting it equal to zero:
```math
\frac{df}{dp_\theta(H)}=\frac{4}{p_\theta(H)}-\frac{2}{1-p_\theta(H)}=0\Rightarrow 4\left(1-p_\theta(H)\right)=2p_\theta(H)\Rightarrow p_\theta(H)=\frac{4}{6}
```

### 3-2. Detailed Derivation of ELBO
A more detailed derivation of ELBO (see [Luo et al. (2022), "Background: ELBO, VAE, and Hierarchical VAE"](#4-2-understanding-diffusion-models-a-unified-perspective)) is:

```math
\begin{align}
\log p_\theta(x)&=\log p_\theta(x)\int q_\phi(z\mid x)dz \\
&=\int q_\phi(z\mid x)\log p_\theta(x)dz \\
&=\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x)\right] \\
&=\mathbb{E}_{q_\phi(z\mid x)}\left[\log\frac{p_\theta(x,z)q_\phi(z\mid x)}{p(z\mid x)q_\phi(z\mid x)}\right] \\
&=\mathbb{E}_{q_\phi(z\mid x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]+\mathbb{E}_{q_\phi(z\mid x)}\left[\log\frac{q_\phi(z\mid x)}{p(z\mid x)}\right] \\
&=\mathbb{E}_{q_\phi(z\mid x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right]+D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p(z\mid x)\right) \\
&\ge\mathbb{E}_{q_\phi(z\mid x)}\left[\log\frac{p_\theta(x,z)}{q_\phi(z\mid x)}\right] \\
&=\mathbb{E}_{q_\phi(z\mid x)}\left[\log\frac{p_\theta(x\mid z)p(z)}{q_\phi(z\mid x)}\right] \\
&=\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]-\mathbb{E}_{q_\phi(z\mid x)}\left[\log\frac{q_\phi(z\mid x)}{p(z)}\right] \\
&=\mathbb{E}_{q_\phi(z\mid x)}\left[\log p_\theta(x\mid z)\right]-D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p(z)\right) \\
&=ELBO
\end{align}
```

$`D_{\mathrm{KL}}\left(q_\phi(z\mid x)\,\|\,p(z\mid x)\right)`$ is the term hidden in the derivation of ELBO at section ["1-4. Evidence lower bound (ELBO)"](#1-4-evidence-lower-bound-elbo), and it measures how well the encoder $q_\phi(z\mid x)$ approximates the true posterior $p(z\mid x)$. Maximizing ELBO is equivalent to minimizing this term because their sum $\log p_\theta(x)$ is a constant with respect to $\phi$, as it depends on $\theta$.

### 3-3. Use a Bernoulli Distribution as the Decoder
A Bernoulli distribution $\mathcal{B}(\cdot;\pi)$ has two outcomes $`\{0,1\}`$ and is parameterized by $\pi$ - the probability of $1$ (with the probability of $0$ being $1-\pi$). We can map the latent variable $z$ to $\pi$ using a decoder network $f_\theta(z)$ (similar to how we mapped $z$ to the Gaussian mean in section ["1-4-1. More About the Reconstruction Term"](#1-4-1-more-about-the-reconstruction-term)).

```math
\begin{align}
\log p_\theta(x\mid z)&=\log \mathcal{B}(x;f_\theta(z)) \\
&=\log\left((f_\theta(z))^x(1-f_\theta(z))^{1-x}\right) \\
&=x\log f_\theta(z)+(1-x)\log\left(1-f_\theta(z)\right)
\end{align}
```

Adding a minus sign $-$ turns this into a binary cross-entropy loss, and you can verify that it is optimized when $f_\theta(z)=x$, just as in the Gaussian case (hint: solve by setting the derivative to zero, similar to ["Appendix 3-1."](#3-1-solution-to-the-coin-toss-example)).

## 4. References
#### 4-1. An Introduction to Variational Autoencoders
[Kingma et al. (2019)](https://arxiv.org/abs/1906.02691)

#### 4-2. Understanding Diffusion Models: A Unified Perspective
[Luo et al. (2022)](https://arxiv.org/abs/2208.11970)

#### 4-3. Tutorial on Diffusion Models for Imaging and Vision
[Chan et al. (2024)](https://arxiv.org/abs/2403.18103)
