abstract type AbstractMINERVA end

"""
    MINERVA(;L=.8, n_features=10, n_traces=20, c=[0.0])

Model object for the MINERVA model model. 

Parameters
- `L`: learning rate or encoding accuracy probability
* `memory`: a matrix containing memory traces where each row corresponds to a trace
* `n_features`: the number of features per trace
* `n_traces`: the number of traces in model
* `c`: a vector of decision criteria

**References**

Hintzman, D. L. (1988). Judgments of frequency and recognition model in a multiple-trace model model. 
    Psychological Review, 95(4), 528.
"""
mutable struct MINERVA <: AbstractMINERVA
    L::Float64
    memory::Array{Int,2}
    n_features::Int
    n_traces::Int
    c::Vector{Float64}
end

Broadcast.broadcastable(x::MINERVA) = Ref(x)

function MINERVA(;L=.8, n_features=10, n_traces=20, c=[0.0])
    memory = fill(0, n_traces, n_features)
    return MINERVA(L, memory, n_features, n_traces, c)
end

function stimulus_similarity(model, p)
     rand() <= model.h ? (return p) : (return rand(setdiff(-1:1, p)))
end

"""
    random_stimulus(model::AbstractMINERVA)

Generates random stimulus with a specified number of features in the `model` object. 
"""
random_stimulus(model::AbstractMINERVA) = random_stimulus(model.n_features)
random_stimulus(n_features) = rand(-1:1, n_features)

"""
    random_stimulus(model::AbstractMINERVA, n)

Generates `n` random stimuli with a specified number of features in model object. Returns a `n` by 
`n_features` matrix. 
"""
random_stimulus(model::AbstractMINERVA, n) = random_stimulus(n, model.n_features)
random_stimulus(n_traces, n_features) = rand(-1:1, n_traces, n_features)


random_distractor(model::AbstractMINERVA) = random_stimulus(model::AbstractMINERVA)

function encode(model::AbstractMINERVA, x)
    x == 0 ? (return 0) : nothing
    rand() <= model.L ? (return x) : (return 0)
end

"""
    encode!(model, stimuli)

Encodes a matrix of stimuli with accuracy governed by parameter `L` 

* `model`: `MINERVA` model object 
* `stimuli`: a matrix in which rows represent stimuli and columns represent features

"""
function encode!(model::AbstractMINERVA, stimuli)
    model.memory .= encode.(model, stimuli)
    return nothing
end

"""
    compute_intensity(model::AbstractMINERVA, probe)

Computes echo intensity using a given memory `probe`. 

* `model`: `MINERVA` model object 
* `probe`: a vector of feature values representing a memory probe

"""
compute_intensity(model::AbstractMINERVA, probe) = compute_intensity(model.memory, probe)

function compute_intensity(memory, probe)
    return sum(compute_activation(memory, probe))
end

"""
    compute_activation(M, p)

Computes activation value for each trace in matrix `M` for a given probe `p`. 

* `M`: a matrix in which rows represent memory traces and columns represent feature values
* `p`: a vector of feature values representing a memory probe

"""
compute_activation(M, p) = compute_similarity(M, p) .^ 3

"""
    compute_activation(M, p)

Computes simularity between memory traces in matrix `M` and memory probe `p` using the dot product. 

* `M`: a matrix in which rows represent memory traces and columns represent feature values
* `p`: a vector of feature values representing a memory probe
"""
function compute_similarity(M, p)
    s = M * p
    n = length(p) .- sum((M .== 0) * (p .== 0), dims=2)
    return s ./ n
end


