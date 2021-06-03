using Cubature
using Distributions
using LaTeXStrings
using LinearAlgebra
using Plots

include("src/ais_rb.jl")
include("src/ais_ob.jl")

function init_test(d)
    μ_d = rand(d)
    A = rand(d,d)
    ## Trick to make Σ_d positive semidefinite
    Σ_d = A'*A
    f = x -> pdf(MvTDist(2, μ_d, Σ_d), x)
    return μ_d, Σ_d, f
end

function run_ais_rb_normal_test(d::Int, n_iters_test::Int, n_stages_rb::Int)
    error_list_rb_final = zeros(n_stages_rb)
    for i in 1:n_iters_test
        @time est, error_list = ais_rb(d, n_stages_rb, f, 0.01, μ_d)
        error_list_rb_final = error_list_rb_final.*(i-1)/i + error_list./i
    end
    return error_list_rb_final
end

function run_ais_ob_normal_test(d::Int, n_iters_test::Int, n_stages::Int)
    error_list_ob_final = zeros(n_stages_ob)
    for i in 1:n_iters_test
        @time est, error_list = ais_ob(d, n_stages_ob, f, μ_d, n_samples_ob)
        error_list_ob_final = error_list_ob_final.*(i-1)/i + error_list./i
    end
    return error_list_ob_final
end

## Initialize integrand and test parameters
d = 2
μ_d, _, f = init_test()
n_iters_test = 10

## RB tests
n_stages_rb = 500000
error_list_rb_final = run_ais_rb_normal_test(d, n_iters_test, n_stages_rb)

plot_gap = 10
plot(log10.([k for k in Int(n_stages_rb/2):plot_gap:n_stages_rb]), log10.(error_list_rb_final[Int(n_stages_rb/2):plot_gap:end]),
     xlabel = L"$\log(N_s)$", ylabel = L"$\log(\varepsilon_{rel})$", label = "Stochastic Convex Programming")

## OB tests
n_stages_ob = 2000
n_samples_ob = [Int(n_stages_ob/2) for k in 1:n_stages_ob]
error_list_ob_final = run_ais_ob_normal_test(d, n_iters_test, n_stages_ob)

 plot_gap = 10
 plot(log10.([k for k in Int(n_stages_ob/2):plot_gap:n_stages_ob]), log10.(error_list_ob_final[Int(n_stages_ob/2):plot_gap:end]),
      xlabel = L"$\log(N_s)$", ylabel = L"$\log(\varepsilon_{rel})$", label = "Cumulative Average Parameter Update")
