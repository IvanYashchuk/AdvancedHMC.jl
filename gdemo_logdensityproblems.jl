using TransformVariables, LogDensityProblems, Distributions

struct GaussianDemo{T,P}
    y::T
    x::T
    s_prior::P
end

function (p::GaussianDemo)((m, s))
    d = Normal(m, √s)
    logpdf(d, p.x) + logpdf(d, p.y) + logpdf(p.s_prior, s) + logpdf(Normal(0, √s), m)
end

p = GaussianDemo(1.5, 2.0, InverseGamma(2, 3))
P = TransformedLogDensity(as((m = asℝ, s = asℝ₊)), p)
∇P = ADgradient(:ForwardDiff, P)

##
## DynamicHMC
##
using DynamicHMC
@time chain, tuned = NUTS_init_tune_mcmc(∇P, 10_000)

mean(get_position.(chain))
#  2-element Array{Float64,1}:
#  1.1859183007401242
#  0.48246364582230816

##
## AdvancedHMC
##

# logdensity(LogDensityProblems.ValueGradient, ∇P, [1.; 1.])
function ℓπ(θ)
    res = logdensity(LogDensityProblems.Value, ∇P, θ)
    return res.value
end

function ∂ℓπ∂θ(θ)
    res = logdensity(LogDensityProblems.ValueGradient, ∇P, θ)
    return (res.value, res.gradient)
end


using AdvancedHMC

D = 2

# Sampling parameter settings
n_samples = 10_000
n_adapts = 2_000

# Draw a random starting points
θ_init = randn(D)

# Define metric space, Hamiltonian, sampling method and adaptor
# m = [0.5, 0.25]; metric = DiagEuclideanMetric(m .* m, m, zeros(2))
metric_diag = DiagEuclideanMetric(D)
metric_dense = DenseEuclideanMetric(D)

h_diag = Hamiltonian(metric_diag, ℓπ, ∂ℓπ∂θ)
h_dense = Hamiltonian(metric_dense, ℓπ, ∂ℓπ∂θ)

init_eps = Leapfrog(find_good_eps(h_diag, θ_init)) # diag or dense both work.
prop_nuts = AdvancedHMC.NUTS(init_eps; sampling=:slice)
prop_hmcda = AdvancedHMC.HMCDA(init_eps, 0.5)

da_adaptor = NesterovDualAveraging(0.8, init_eps.ϵ)
stan_adaptor_diag = StanHMCAdaptor(n_adapts, Preconditioner(metric_diag), da_adaptor)
stan_adaptor_dense = StanHMCAdaptor(n_adapts, Preconditioner(metric_dense), da_adaptor)

# Draw samples via simulating Hamiltonian dynamics
# - `samples` will store the samples
# - `stats` will store statistics for each sample
samples_hmcda, stats = sample(h_diag, prop_hmcda, θ_init, n_samples, da_adaptor, n_adapts);
samples_nuts, stats = sample(h_diag, prop_nuts, θ_init, n_samples, da_adaptor, n_adapts);

samples_hmcda_mass, stats = sample(h_diag, prop_hmcda, θ_init, n_samples, stan_adaptor_diag, n_adapts);
samples_nuts_mass, stats = sample(h_dense, prop_nuts, θ_init, n_samples, stan_adaptor_dense, n_adapts);

mean(samples_hmcda)
mean(samples_nuts)
# 2-element Array{Float64,1}:
# 1.1642886419329694
# 0.4892413609447907


mean(samples_hmcda_mass)
mean(samples_nuts_mass)
# 2-element Array{Float64,1}:
# 1.1630877430250832
# 0.19589947526414722
