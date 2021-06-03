"""
Implmentation of the Monte Carlo Integrator with Adaptive Importance Sampling
from Ryu, E. and Boyd, S., Adaptive Importance Sampling via Stochastic Convex Programming,
arXiv preprint, 2015.

Assumes a functional form f : Rᵈ -> R.
Parameters:
- d: input dimension
- n_stages: number of iterations performed by the integrator
- f: function to be integrated
- C: update step
"""

function ais_rb(d::Int, n_stages::Int, f::Function, C::AbstractFloat, μ_glob::AbstractVector)
    g_lam = (μ, Σ) -> MultivariateNormal(μ, Σ)

    μ = rand(d)
    Σ = I + zeros(d, d)

    m = inv(Σ)*μ
    S = inv(Σ)

    error_list = []
    est = zeros(d)
    for k in 1:n_stages
        μ = inv(S)*m
        X_k = rand(g_lam(μ, Σ))
        est = (est*(k-1) + f(X_k)*X_k/pdf(g_lam(μ, Σ), X_k))/k
        m -= C*(norm(X_k)^2)*(f(X_k)^2)/(2*pdf(g_lam(μ, Σ), X_k)*sqrt(k))*(inv(S)*m-X_k)
        push!(error_list, norm(est-μ_glob)/norm(μ_glob))
    end
    return est, error_list
end
