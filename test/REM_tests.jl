@safetestset "REM" begin
    using GlobalMatchingModels, Statistics, Test, Random

    probe = [6,1,1,3];
    memory = [0 2; 1 2; 0 1; 3 0];
    model = REM(;memory, g=.40, c=.70)
    activations = compute_activations(model, probe)
    @test activations ≈ [10.5802,.18450] atol = 1e-4

    prob = compute_prob(activations)
    @test prob ≈ .8433 atol = 1e-4

    # is there a distribution we can test encode! against?
    Random.seed!(58)
    model = REM(;memory=fill(0, 4, 4), g=.40, c=.70)
    stimuli = generate_stimuli(.3, 4, 4)
    encode!(model, stimuli)
    dims = size(model.memory)
    @test dims == (4,4)
    @test sum(model.memory .!= stimuli) > 0
end
