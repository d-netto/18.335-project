"""
Implmentation of the Monte Carlo Integrator with Adaptive Importance Sampling
from  Oh, M-S and Berger, J., Adaptive importance sampling in Monte Carlo integration, Journal
of Statistical Computation and Simulation, 1992.

Assumes a functional form f : Rᵈ -> R.
Parameters:
- d: input dimension
- n_stages: number of iterations performed by the integrator
- f: function to be integrated
- n_samples: list of number of samples used for each iteration
"""

function ais_ob(d::Int, n_stages::Int, f, μ_glob::AbstractVector, n_samples)
    g_lam = (λ, Σ) -> MultivariateNormal(λ, Σ)

    λ = rand(d)
    Σ = I + zeros(d, d)

    WΦ = zeros(n_stages, d+1)

    error_list = []
    for k in 1:n_stages
        w_k = θ -> f(θ)/pdf(g_lam(λ, Σ), θ)
        samples_k_list = [rand(g_lam(λ, Σ)) for n in 1:n_samples[k]]
        W_k = h -> sum((h.(samples_k_list)).*(w_k.(samples_k_list)))
        WΦ[k, 1] = W_k(x -> one(typeof(x[1])))
        WΦ[k, 2:end] = [W_k(x -> x[k-1]) for k in 2:d+1]
        sum_weights = sum(WΦ[:,1])
        λ = [sum(WΦ[:,k])/sum_weights for k in 2:d+1]
        push!(error_list, norm(λ-μ_d)/norm(μ_d))
    end
    return λ, error_list
end
