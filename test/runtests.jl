using SafeTestsets

@safetestset "MINERVA" begin
    using GlobalMatchingModels, Statistics, Test, Random
    M = [1 1 0; 0 -1 0]
    p = [0, 1, 0]

    s = similarity(M, p)
    @test s == [.5,-1]

    a = activation(M, p)
    @test a == [.125,-1]

    x = compute_intensity(M, p)
    @test x == -0.875

    include("../Examples/MINERVA_Functions.jl")
    Random.seed!(1125435)
    # initialize model
    model = MINERVA(n_features=20, n_traces=60, L=.50)
    # simulate experiment
    sim_data = mapreduce(x->simulate(model, 4, 5), hcat, 1:20000)'
    means = mean(sim_data, dims=1)
    @test means ≈ [.0 0.153 0.31  0.46  0.61  0.77] atol = .01
end
