using Distributions: logpdf, InverseGamma, Normal
using DiffResults: GradientResult, value, gradient
using ForwardDiff: gradient!

function ℓπ(θ)
    s, m = exp(θ[1]), θ[2]
    logprior = logpdf(InverseGamma(2, 3), s) + log(s) + logpdf(Normal(0, sqrt(s)), m)
    loglikelihood = logpdf(Normal(m, sqrt(s)), 1.5) + logpdf(Normal(m, sqrt(s)), 2.0)
    return logprior + loglikelihood
end

function ∂ℓπ∂θ(θ)
    res = GradientResult(θ)
    gradient!(res, ℓπ, θ)
    return (value(res), gradient(res))
end

using AdvancedHMC

D = 2

# Sampling parameter settings
n_samples = 10_000
n_adapts = 2_000

# Draw a random starting points
θ_init = randn(D)

# Define metric space, Hamiltonian, sampling method and adaptor
m = [0.5, 0.25]
metric = DiagEuclideanMetric(m .* m, m, zeros(2))
h = Hamiltonian(metric, ℓπ, ∂ℓπ∂θ)
init_eps = Leapfrog(find_good_eps(h, θ_init))
prop_nuts = NUTS(init_eps; sampling=:slice)
prop_hmcda = HMCDA(init_eps, 0.5)
adaptor = NesterovDualAveraging(0.8, prop.integrator.ϵ)

# Draw samples via simulating Hamiltonian dynamics
# - `samples` will store the samples
# - `stats` will store statistics for each sample
samples_hmcda, stats = sample(h, prop_hmcda, θ_init, n_samples, adaptor, n_adapts)
samples_nuts, stats = sample(h, prop_nuts, θ_init, n_samples, adaptor, n_adapts)
