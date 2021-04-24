#####################################################################################################
#                                         Load Packages and Functions
#####################################################################################################
cd(@__DIR__)
using Pkg
Pkg.activate("..")
using GlobalMatchingModels, Plots, Random
include("MINERVA_Functions.jl")
Random.seed!(356355)
#####################################################################################################
#                                        Specify  Model and Simulate Experiment
#####################################################################################################
# initialize model
model = MINERVA(h=.75, n_features=20, n_traces=60, L=.50)
# simulate experiment
sim_data = mapreduce(x->simulate(model, 4, 5), hcat, 1:10000)'
#####################################################################################################
#                                      Plot Simulation
#####################################################################################################
pyplot()
histogram(sim_data, alpha=.6, grid=false, norm=true, legendtitle="frequency", label=[0:5;]',
    xaxis=font(12), yaxis=font(12), xlabel="Echo Intensity", ylabel="Density", size=(600,300))
savefig("MINERVA.png")

# function noise_distribution(model, stimuli, probe, n_sim)
#     intensity = fill(0.0, n_sim)
#     for i in 1:n_sim
#         encode!(model, stimuli)
#         intensity[i] = compute_intensity(model, probe)
#     end
#     return intensity
# end

# function signal_distribution(model, stimuli, probe, n_sim)
#     intensity = fill(0.0, n_sim)
#     for i in 1:n_sim
#         encode!(model, stimuli)
#         intensity[i] = compute_intensity(model, probe)
#     end
#     return intensity
# end

# model = MINERVA(h=.75, n_features=30, n_traces=20, L=.50)

# stimuli = generate_stimuli(model)



# target_probe = random_target(stimuli)
# intensity = signal_distribution(model, stimuli, target_probe, 10_000)
# println(mean(intensity), " ", std(intensity))

# distractor_probe = random_distractor(model)
# intensity = signal_distribution(model, stimuli, target_probe, 10_000)
# println(mean(intensity), " ", std(intensity))



# function random_target(stimuli)
#     idx = rand(1:size(stimuli, 1))
#     return stimuli[idx,:]
# end

# function generate_stimuli(model)
#     @unpack n_traces, n_features = model
#     prototype = random_distractor(n_features)
#     stimuli = fill(0, n_traces, n_features)
#     for t in 1:n_traces, f in 1:n_features
#         stimuli[t,f] = stimulus_similarity(model, prototype[f])
#     end
#     return stimuli
# end
