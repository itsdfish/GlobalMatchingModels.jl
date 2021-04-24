module GlobalMatchingModels
    using Distributions, Parameters, Statistics
    export activation, compute_intensity, similarity, encode, encode!, random_stimulus
    export MINERVA
    
    include("MINERVA.jl")
end
