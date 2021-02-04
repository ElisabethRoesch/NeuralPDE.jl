using Flux, Zygote, StochasticDiffEq
using LinearAlgebra, Statistics
println("heat neural SDE")
using Test, NeuralPDE

using Random
Random.seed!(100)



println("high-dimensional heat equation")
d = 100 # number of dimensions
x0 = fill(8.0f0,d)
tspan = (0.0f0,2.0f0)
dt = 0.5
m = 50 # number of trajectories (batch size)

g(X) = sum(X.^2)
f(X,u,σᵀ∇u,p,t) = Float32(0.0)
μ_f(X,p,t) = zero(X) #Vector d x 1
σ_f(X,p,t) = Diagonal(ones(Float32,d)) #Matrix d x d
prob = TerminalPDEProblem(g, f, μ_f, σ_f, x0, tspan)

hls = 10 + d #hidden layer size
#sub-neural network approximating solutions at the desired point
u0 = Flux.Chain(Dense(d,hls,relu),
                Dense(hls,hls,relu),
                Dense(hls,1))
# sub-neural network approximating the spatial gradients at time point
σᵀ∇u = Flux.Chain(Dense(d+1,hls,relu),
                  Dense(hls,hls,relu),
                  Dense(hls,d))
opt = Flux.ADAM(0.005)  #optimizer
pdealg = NNPDENS(u0, σᵀ∇u, opt=opt)

ans = solve(prob, pdealg, verbose=true, maxiters=500, trajectories=m,
                            alg=EM(), dt=dt, pabstol = 1f-6)

u_analytical(x,t) = sum(x.^2) .+ d*t
analytical_ans = u_analytical(x0, tspan[end])
