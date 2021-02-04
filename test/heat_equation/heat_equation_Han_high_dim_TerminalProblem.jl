using Flux, Zygote, LinearAlgebra, Statistics
println("Heat han")
using Test, StochasticDiffEq

using NeuralPDE

using Random
Random.seed!(100)

opt = Flux.ADAM(0.005)  #optimizer
# high-dimensional heat equation
d = 30 # number of dimensions
x0 = fill(8.0f0,d)
tspan = (0.0f0,2.0f0)
dt = 0.5
time_steps = div(tspan[2]-tspan[1],dt)
m = 30 # number of trajectories (batch size)

g(X) = sum(X.^2)
f(X,u,σᵀ∇u,p,t) = 0.0
μ_f(X,p,t) = 0.0
σ_f(X,p,t) = 1.0
prob = TerminalPDEProblem(g, f, μ_f, σ_f, x0, tspan)


hls = 10 + d #hidden layer size
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = [Flux.Chain(Dense(d,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d)) for i in 1:time_steps]

alg = NNPDEHan(u0, σᵀ∇u, opt = opt)

ans = solve(prob, alg, verbose = true, abstol=1e-8, maxiters = 1000, dt=dt, trajectories=m)

u_analytical(x,t) = sum(x.^2) .+ d*t
analytical_ans = u_analytical(x0, tspan[end])
error_l2 = sqrt((ans - analytical_ans)^2/ans^2)

println("high-dimensional heat equation")
# println("numerical = ", ans)
# println("analytical = " ,analytical_ans)
println("error_l2 = ", error_l2, "\n")
@test error_l2 < 1.0
