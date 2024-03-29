# GlobalMatchingModels

This package will feature a variety of global matching models of recognition memory. Currently, the only models available are MINERVA 2 and REM. More to follow. 

## MINERVA 2

MINERVA 2 is a global matching model of recognition memory and frequency judgment. According to MINERVA 2, each experienced event is stored as a vector representing feature values. Feature values can take the 
value -1, which is inhibatory, 0 which indicates an absence of information, or 1, which is excitatory. Event information is encoded into a long term memory store. Feature values are encoded accurately with probability `L`. Recognition memory and frequency judgments are based on the similarity between a memory probe or cue and stored traces in long term memory. 

### Example 

The following code illustrates how frequency of experience increases the echo intensity which is a measure
of memory "signal". During each simulated experiment, the model encodes 20 unique events into memory. Four events are encoded once, four events are encoded twice, and so on, until four events are encoded five times, resulting in 60 learning memory traces. The echo intensity distribution will be ploted for each frequency distribution.

First, we will load the required packages and code. You will have to install `StatsPlots` in your global environment. 
```julia
cd(@__DIR__)
using Pkg
Pkg.activate("..")
using GlobalMatchingModels, StatsPlots, Random
include("MINERVA_Functions.jl")
Random.seed!(356355)
```

Next, a model object with 60 memory traces and 20 features is generated. Encoding accuracy parameter `L`
is set to .5. The next line of code simulates the model 10,000 times. On each simulation, it generates four random event vectors for each of the zero to five frequency conditions.  

```julia
# initialize model
model = MINERVA(n_features=20, n_traces=60, L=.50)
# simulate experiment
sim_data = mapreduce(x->simulate(model, 4, 5), hcat, 1:10000)'
```

The plot replicates the results reported in Figure 4 of Hintzman (1988), showing that the echo intensity distributions increase in mean and variance as a function of frequency.

```julia
pyplot()
density(sim_data, alpha=.9, grid=false, norm=true, legendtitle="Frequency", label=[0:5;]',
    legendtitlefontsize=9, xaxis=font(12), yaxis=font(12), xlabel="Echo Intensity", ylabel="Density", 
    linewidth=1.5, bins=50, size=(600,300))
```
<img src="Examples/MINERVA.png" alt="" width="600" height="300">


*References*

1. Hintzman, D. L. (1988). Judgments of frequency and recognition model in a multiple-trace model model. 
    Psychological Review, 95(4), 528.

*Authors*

1. [itsdfish](https://github.com/itsdfish)


## REM

REM (Retrieving Effectively from Memory) is a global matching model of recognition memory. The model assumes that each memory is stored as an "image" of feature values drawn from a geometric distribution. Errors occur during encoding, which results in a degraded memory. The accuracy of the encoded memory representation is controled by parameters `g` and `u`. Use `?` to see documentation for details. 

### Example

The following example is taken from Figure 1 of Shiffrin & Strevers (1997).

```julia
probe = [6,1,1,3];
memory = [0 2; 1 2; 0 1; 3 0];
model = REM(;memory, g=.40, c=.70)
activations = compute_activations(model, probe)
odds = compute_odds(activations)
````
which returns the following result:

```julia
julia> odds = compute_odds(activations)
5.3823888888888884
```
*References*

1. Shiffrin, R. M., & Steyvers, M. (1997). A model for recognition memory:
    REM—Retrieving Effectively From Memory. Psychonomic Bulletin & Review,
    4(2), 145-166.

*Authors*

1. [tmc2737](https://github.com/tmc2737)
2. [itsdfish](https://github.com/itsdfish)
