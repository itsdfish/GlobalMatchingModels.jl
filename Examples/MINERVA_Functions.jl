repeat_stimulus(x, n_rep) = repeat(x, inner=(n_rep,1))

function simulate(model, n_type, max_rep)
    # generate random stimuli grouped by frequency of presentation
    stimuli = [random_stimulus(model, n_type) for _ in 0:max_rep]
    # study list in which stimuli are repeated 1 through max_rep times depending on group
    study_list = mapreduce(i->repeat_stimulus(stimuli[i+1], i), vcat, 1:max_rep)
    # encode study list
    encode!(model, study_list)
    # compute echo intensity for each frequency group
    return map(s->compute_intensity(model, s[1,:]), stimuli)
end