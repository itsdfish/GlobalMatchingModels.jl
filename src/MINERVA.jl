"""
    MINERVA(;L=.8, s=.9, N=10, S=20, h=.8, c=0)

Model object for the MINERVA model model. 

Parameters
- `L`: learning rate or encoding accuracy probability
* `s`:
* `memory`: a matrix containing memory traces where each row corresponds to a trace
* `n_features`: the number of features per trace
* `n_traces`: the number of traces in model
* `h`:
* `c`:

**References**

Hintzman, D. L. (1988). Judgments of frequency and recognition model in a multiple-trace model model. 
    Psychological Review, 95(4), 528.
"""
mutable struct MINERVA
    L::Float64
    memory::Array{Int,2}
    n_features::Int
    n_traces::Int
    h::Float64
    c::Float64
end

Broadcast.broadcastable(x::MINERVA) = Ref(x)

function MINERVA(;L=.8, n_features=10, n_traces=20, h=.8, c=0)
    memory = fill(0, n_traces, n_features)
    return MINERVA(L, memory, n_features, n_traces, h, c)
end

function stimulus_similarity(model, p)
     rand() <= model.h ? (return p) : (return rand(setdiff(-1:1, p)))
end

random_stimulus(model::MINERVA) = random_stimulus(model.n_features)
random_stimulus(n_features) = rand(-1:1, n_features)

random_stimulus(model::MINERVA, n) = random_stimulus(n, model.n_features)
random_stimulus(n, n_features) = rand(-1:1, n, n_features)

random_distractor(model::MINERVA) = random_stimulus(model::MINERVA)

function encode(model, x)
    x == 0 ? (return 0) : nothing
    rand() <= model.L ? (return x) : (return 0)
end

function encode!(model, stimuli)
    model.memory .= encode.(model, stimuli)
    return nothing
end

function compute_intensity(model, probe)
    return sum(activation(model.memory, probe))
end

activation(M, p) = similarity(M, p) .^ 3

function similarity(M, p)
    s = M * p
    n = length(p) .- sum((M .== 0) * (p .== 0), dims=2)
    return s ./ n
end


