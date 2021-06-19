@safetestset "REM" begin
    using GlobalMatchingModels, Statistics, Test, Random
    using Distributions 
    probe = [6,1,1,3];
    memory = [0 2; 1 2; 0 1; 3 0];
    model = REM(;memory, g=.40, c=.70)
    activations = compute_activations(model, probe)
    @test activations ≈ [10.5802,.18450] atol = 1e-4

    prob = compute_prob(activations)
    @test prob ≈ .8433 atol = 1e-4

    Random.seed!(58)
    model = REM(;memory=fill(0, 4, 4), g=.40, c=.70)
    stimuli = generate_stimuli(.3, 4, 4)
    encode!(model, stimuli)
    dims = size(model.memory)
    @test dims == (4,4)
    @test sum(model.memory .!= stimuli) > 0

    Random.seed!(25)
    g = .30
    c = .70
    u = .80
    n = 5000
    stimuli = generate_stimuli(g, n, n)
    model = REM(;memory=fill(0, n, n), g, c, u)
    encode!(model, stimuli)
    est = mean(model.memory .== 0)
    @test est ≈  (1 - u) + u * g * c + u * (1 - c) * g atol = 1e-4
end
