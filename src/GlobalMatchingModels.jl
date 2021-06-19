module GlobalMatchingModels
    using Distributions, Parameters, Statistics
    export compute_activation, compute_intensity, compute_similarity, encode, encode!, random_stimulus
    export compute_activations, compute_prob, compute_odds, generate_stimuli
    export MINERVA, REM
    
    include("MINERVA.jl")
    include("REM.jl")
end
