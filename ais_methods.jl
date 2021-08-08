using Cubature
using Distributions
using LaTeXStrings
using LinearAlgebra
using Plots

"""
Implmentation of the Monte Carlo Integrator with Adaptive Importance Sampling
from Ryu, E. and Boyd, S., Adaptive Importance Sampling via Stochastic Convex Programming,
arXiv preprint, 2015.

Assumes a functional form f : Rᵈ -> R.
Parameters:
- d: input dimension
- n_stages: number of iterations performed by the integrator
- f: pdf whose expectation should be integrated
- C: update step
- μ_glob: mean of f
"""
function ais_rb(
    d::Int,
    n_stages::Int,
    f::Function,
    C::AbstractFloat,
    μ_glob::AbstractVector,
)
    g_lam = (μ, Σ) -> MultivariateNormal(μ, Σ)

    μ = rand(d)
    Σ = I + zeros(d, d)

    m = inv(Σ) * μ
    S = inv(Σ)

    X_k = zeros(d)
    est = zeros(d)

    error_list = Float64[]

    @inbounds for k in 1:n_stages
        mul!(μ, inv(S), m)
        X_k .= rand(g_lam(μ, Σ))
        est .= (est .* (k - 1) .+ f(X_k) .* X_k / pdf(g_lam(μ, Σ), X_k)) / k
        m .-=
            C .* (norm(X_k)^2) .* (f(X_k)^2) ./ (2 * pdf(g_lam(μ, Σ), X_k) * sqrt(k)) *
            (μ - X_k)
        push!(error_list, norm(est .- μ_glob) / norm(μ_glob))
    end
    return est, error_list
end

"""
Implmentation of the Monte Carlo Integrator with Adaptive Importance Sampling
from  Oh, M-S and Berger, J., Adaptive importance sampling in Monte Carlo integration, Journal
of Statistical Computation and Simulation, 1992.

Assumes a functional form f : Rᵈ -> R.
Parameters:
- d: input dimension
- n_stages: number of iterations performed by the integrator
- f: pdf whose expectation should be integrated
- μ_glob: mean of f
- n_samples: list of number of samples used for each iteration
"""
function ais_ob(
    d::Int,
    n_stages::Int,
    f::Function,
    μ_glob::AbstractVector,
    n_samples::AbstractVector,
)
    g_lam = (λ, Σ) -> MultivariateNormal(λ, Σ)

    λ = rand(d)
    Σ = I + zeros(d, d)

    WΦ = zeros(n_stages, d + 1)
    error_list = Float64[]

    @inbounds for k in 1:n_stages
        w_k = θ -> f(θ) / pdf(g_lam(λ, Σ), θ)
        samples_k_list = [rand(g_lam(λ, Σ)) for n = 1:n_samples[k]]
        W_k = h -> sum((h.(samples_k_list)) .* (w_k.(samples_k_list)))
        WΦ[k, 1] = W_k(x -> one(eltype(x)))
        WΦ[k, 2:end] .= [W_k(x -> x[k-1]) for k = 2:d+1]
        sum_weights = sum(@view WΦ[:, 1])
        λ = [sum(@view WΦ[:, k]) ./ sum_weights for k = 2:d+1]
        push!(error_list, norm(λ - μ_d) / norm(μ_d))
    end
    return λ, error_list
end

function init_test(d::Int)
    μ_d = rand(d)
    A = rand(d, d)
    ## Trick to make Σ_d positive semidefinite
    Σ_d = A' * A
    f = x -> pdf(MvTDist(2, μ_d, Σ_d), x)
    return μ_d, Σ_d, f
end

function run_ais_rb_normal_test(d::Int, n_iters_test::Int, n_stages_rb::Int)
    error_list_rb_final = zeros(n_stages_rb)
    for i = 1:n_iters_test
        @time est, error_list = ais_rb(d, n_stages_rb, f, 0.01, μ_d)
        error_list_rb_final = error_list_rb_final .* (i - 1) / i + error_list ./ i
    end
    return error_list_rb_final
end

function run_ais_ob_normal_test(d::Int, n_iters_test::Int, n_stages::Int, n_samples_ob::AbstractVector)
    error_list_ob_final = zeros(n_stages_ob)
    for i = 1:n_iters_test
        @time est, error_list = ais_ob(d, n_stages_ob, f, μ_d, n_samples_ob)
        error_list_ob_final = error_list_ob_final .* (i - 1) / i + error_list ./ i
    end
    return error_list_ob_final
end

# Initialize integrand and test parameters
d = 2
μ_d, _, f = init_test(d)
n_iters_test = 10

# RB tests
n_stages_rb = 1000000
error_list_rb_final = run_ais_rb_normal_test(d, n_iters_test, n_stages_rb)

plot_gap = 10
plot(
    log10.([k for k = Int(n_stages_rb / 2):plot_gap:n_stages_rb]),
    log10.(error_list_rb_final[Int(n_stages_rb / 2):plot_gap:end]),
    xlabel = L"$\log(N_s)$",
    ylabel = L"$\log(\varepsilon_{rel})$",
    label = "Stochastic Convex Programming",
)

# OB tests
n_stages_ob = 1000
n_samples_ob = [Int(n_stages_ob / 2) for k = 1:n_stages_ob]
error_list_ob_final = run_ais_ob_normal_test(d, n_iters_test, n_stages_ob, n_samples_ob)

plot_gap = 10
plot(
    log10.([k for k = Int(n_stages_ob / 2):plot_gap:n_stages_ob]),
    log10.(error_list_ob_final[Int(n_stages_ob / 2):plot_gap:end]),
    xlabel = L"$\log(N_s)$",
    ylabel = L"$\log(\varepsilon_{rel})$",
    label = "Cumulative Average Parameter Update",
)
