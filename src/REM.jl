"""
    REM(;memory, u=.9, c=.6, g=.4, n_steps=1)

A REM model object containing a memory array and parameters.
 
# Arguments
- `memory::AbstractArray`: An array of item representations.
- `u::Float64`: Probability (between 0 and 1) that a given feature of an item is
    copied into LTM.
- `c::Float64`: Probability (between 0 and 1) that a given feature of an item is
    copied *correctly* into LTM. 1 - c probability of a feature being converted
    to a different integer sampled from geometric distribution.
- `g::Float64`: Probability of sample integers from a geometric distribution.
- `n_steps::Int64`: The number of timesteps, i.e. number of times an item is studied.

*References*

Shiffrin, R. M., & Steyvers, M. (1997). A model for recognition memory:
    REM—Retrieving Effectively From Memory. Psychonomic Bulletin & Review,
    4(2), 145-166.

* Authors *
@author taylor "dot" curley@gatech.edu (tmc2737)
@date 06/17/2021
@coauthor itsdfish 
@date 06/18/2021
"""
mutable struct REM{T}
    memory::T
    u::Float64
    c::Float64
    g::Float64
    n_steps::Int
end

REM(;memory, u=.9, c=.6, g=.4, n_steps=1) = REM(memory, u, c, g, n_steps)

"""
    generate_stimuli(g, w, n)

Generates a matrix containing individual
vectors of feature values corresponding to different items. Rows correspond to feature
values and columns correspond to memory traces or "images".

# Arguments
- `g::Float64`: Controls the probability of sampling integers from a geometric
    distribution (between 0 and 1). A higher g value represents common words
    (with common features) and lower g values represent words with less common
    words (with uncommon features).
- `w::Int64`: The number of non-zero values in a given item vector.
- `n::Int64`: The number of items in a study/test list.

# Output
- `outMat::AbstractArray`: An w x n matrix of separate item vectors.

# Example
```jldoctest
julia> generate_stimuli(0.4, 4, 4)
4×4 Array{Float64,2}:
 6.0  1.0  0.0  1.0
 0.0  6.0  2.0  1.0
 0.0  0.0  1.0  0.0
 0.0  0.0  3.0  0.0
```
"""
function generate_stimuli(g, w, n)
    return rand(Geometric(g), w, n)
end

"""
    encode!(model::REM, stimuli)

Encodes a degraded copy of stimuli into memory.

# Arguments
- `memory::AbstractArray`: A vector or matrix of item representations. 
- `u::Float64`: Probability (between 0 and 1) that a given feature of an item is
    copied into LTM.
- `c::Float64`: Probability (between 0 and 1) that a given feature of an item is
    copied *correctly* into LTM. 1 - c probability of a feature being converted
    to a different integer sampled from geometric distribution.
- `g::Float64`: Probability of sample integers from a geometric distribution.
- `n_steps::Int64`: The number of timesteps, i.e. number of times an item is studied.

# Output
- `encode!::AbstractArray`: an array of imperfect copy of items.

# Example
```jldoctest
julia> model = REM(;memory=fill(0, 4, 4), g=.40, c=.70)
julia> stimuli = generate_stimuli(.3, 4, 4)
julia> encode!(model, stimuli)
4×4 Matrix{Int64}:
 0  0  4   3
 1  1  0  15
 4  0  0   1
 0  0  0   0
```
"""
function encode!(model::REM, stimuli)
    @unpack memory,u,c,g,n_steps = model
    memory .= stimuli
    memory .= encode_element.(memory, u, c, g, n_steps)
end

function encode_element(element, u, c, g, n_steps)
    new_element = 0
    for i in 1:n_steps
        new_element ≠ 0 ? break : nothing 
        if rand(Uniform(0,1)) < u
            if rand(Uniform(0,1)) < c
                new_element = element
            else
                new_element = rand(Geometric(g))
            end
        end
    end
    return new_element
end

"""
    compute_activations(model::REM, probe)

Computes activations for each trace in memory given a memory probe. Activation estimates based on Equations 3 and 4 from
original paper. The final product of the calculations is a probability ratio of the
probe activating an old item vs. a new item. The example provided is from Figure 1
of the paper.

# Arguments
- `model::REM`: an REM model object containing memory array and parameters
- `probe::AbstractArray`: A vector of numbers representing the memory probe.
- `memory::AbstractArray`: A vector of numbers representing an item in memory.

# Output
- `::Float64`: Float value representing the estimated odds ratio.

# Example
```jldoctest
julia> probe = [6,1,1,3];
julia> memory = [0 2; 1 2; 0 1; 3 0];
julia> model = REM(;memory, g=.40, c=.70)
julia> compute_activations(model, probe)
2-element Vector{Float64}:
 10.580277777777777
  0.18450000000000003
```
"""
function compute_activations(model::REM, probe)
    @unpack memory,g,c = model 
    _,n_cols = size(memory) 
    ratios = fill(0.0, n_cols)
    for i in 1:n_cols
        ratios[i] = compute_activation(memory[:,i], probe, g, c)
    end
    return ratios
end

function compute_activation(memory, probe, g, c)
    prod_ratio = 1.0
    for i in 1:length(memory)
        prod_ratio *= compute_ratio(memory[i], probe[i], g, c)
    end
    return prod_ratio
end

function compute_ratio(m_feature, p_feature, g, c)
    m_feature == 0 ? (return 1.0) : nothing
    m_feature != p_feature ? (return 1 - c) : nothing 
    denom = g * (1 - g)^(m_feature - 1)
    return (c + (1 - c) * denom) / denom
end

"""
    compute_odds(activations)

Given a set of activations, what are the odds that
the probe is old vs. new? An OR > 1 --> old item. The odds 
are computed simply as a mean across activation values.

# Arguments
- `activations::AbstractArray`: A vector of activations.

# Output
- `::Float64`: Float value representing the odds of old vs. new.

# Example
```jldoctest
julia> activations = [10.58,0.18];
julia> compute_odds(activations)
5.38
```
"""
function compute_odds(activations)
    return mean(activations)
end

"""
    compute_prob(activations)

Given a set of activations, what is the probability that
the probe is old vs. new? An prob > .5 --> old item. The probability 
is based on the odds, which is the mean across activations.

# Arguments
- `activations::AbstractArray`: A vector of activations.

# Output
- `::Float64`: Float value representing the odds of old vs. new.

# Example
```jldoctest
julia> activations = [10.58,0.18];
julia> compute_odds(activations)
5.38
```
"""
function compute_prob(activations)
    odds = compute_odds(activations)
    return compute_prob(odds)
end

function compute_prob(odds::Float64)
    return odds / (1 + odds)
end