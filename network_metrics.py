import networkx as nx
import numpy as np

import mentor_matching
from network_analysis import get_pods

### constraints that must be satisfied otherwise a network is discarded
def check_any_mentee_has_no_mentor(people,network):
    """ Count the fraction of people who requested mentors
        but did not receive any mentor"""
    num = 0
    denom = 0
    for person in people.values():
        ## if this person requested any mentors at all
        if (person.n_mentors_total):
            denom+=1
            ## if they got any mentor at all
            if (len(person.mentor_matches)):
                num+=1

    ## return the fraction
    return num == denom

def check_any_matched_avoid(people,network):
    """ Count the fraction of mentors and mentees who
        got matched with people they wanted to avoid
        THIS SHOULD ALWAYS BE 0"""
    num = 0
    for person in people.values():
        broke = False
        for mentee in person.mentee_matches:
            if mentee.name in person.mentees_avoid:
                num+=1
                broke=True
                break
        ## skip checking below if we already had a hit above
        if broke: continue
        for mentor in person.mentor_matches:
            if mentor.name in person.mentors_avoid:
                num+=1
                break

    return num == 0

def check_any_mentors_overassigned(people,network):
    """ Count the fraction of mentors who received
    more mentees than they offered"""
    num = 0
    denom = 0
    for person in people.values():
        ## if this person offered to take any mentees at all
        if (person.n_mentees_total):
            denom+=1
            ## if they got assigned more mentees than they offered
            if (len(person.mentee_matches) - person.n_mentees_total > 0): ##boolean of a negative number is True so need to add > 0
                num+=1
    return num == 0


constraints = [
    check_any_mentee_has_no_mentor,
    check_any_matched_avoid,
    check_any_mentors_overassigned]

### Metrics to optimize:
def run_frac_mentees_less_than_requested(people,network):
    """ Count the fraction of people who requested mentors
        but did not receive as many mentors as requested """
    num = 0
    denom = 0
    for person in people.values():
        ## if this person requested any mentors at all
        if (person.n_mentors_total):
            denom+=1
            ## if they didn't get as many as they requested
            if (person.n_mentors_total - len(person.mentor_matches) > 0):
                num+=1
    ## return the fraction
    return num/denom

def run_frac_mentors_assigned_mentees(people,network):
    """ Count the fraction of mentors who volunteered
        but did not receive any mentees """
    num = 0
    denom = 0
    for person in people.values():
        ## if this person volunteered to mentor
        if (person.n_mentees_total):
            denom+=1
            ## if they got assigned any number of mentees at all
            if (len(person.mentee_matches)):
                num+=1
    ## return the fraction
    return num/denom

def run_frac_mentors_with_extra_slots(people,network):
    """ Count the fraction of mentors who volunteered
        but did not receive as many mentees as they offered"""
    num = 0
    denom = 0
    for person in people.values():
        ## if this person offered to take any mentees at all
        if (person.n_mentees_total):
            denom+=1
            ## if they didn't get as many as they offered
            if (person.n_mentees_total - len(person.mentee_matches) > 0): ##boolean of a negative number is True so need to add > 0
                num+=1
    ## return the fraction
    return num/denom
    
def run_frac_mentees_atleast_one_preference(people,network):
    """ Count the fraction of mentees who received
        at least one mentor they preferred"""
    num = 0
    denom = 0
    for person in people.values():
        ## if this person requested any mentors at all and had preference
        if (person.n_mentors_total and person.mentors_prefr):
            denom+=1
            for mentor in person.mentor_matches:
                ## if they preferenced this mentor
                if (mentor.name in person.mentors_prefr):
                    num+=1
                    break
    ## return the fraction
    return num/denom
    
def run_frac_mentees_alternatives(people,network):
    """ Count the fraction of mentees who
        got matched with alternative mentors
        from different roles than they requested
        This should be minimized"""
    num = 0
    denom = 0
    for person in people.values():
        if (person.n_mentors_total):
            denom+=1
            ## check if for any role a mentee got assigned more mentors than requested. This should be alternatives in that case
            if (np.any(person.n_role_mentors - person.has_n_role_mentors < 0)):
                num += 1
    return num/denom

def run_mean_clique_size(people, network):
    # get the mean clique size for the network.  Presumably this should be maximized

    mean_clique_size = 0
    denom = 0
    for clique in nx.enumerate_all_cliques(network.to_undirected()):
        mean_clique_size += len(clique)
        denom += 1

    if (denom > 0):
        mean_clique_size /= denom

    return mean_clique_size

def get_n_cliques_gtN(network, n = 2):
    # get tne number of cliques with size > N 

    n_cliques = 0
    for clique in nx.enumerate_all_cliques(network.to_undirected()):
        if (len(clique) > n):
            n_cliques += 1

    return n_cliques

def run_n_cliques_gt3(people, network):
    # get tne number of cliques with size > 3 (not sure the best number)

    return get_n_cliques_gtN(network, 3)

def run_n_cliques_gt2(people, network):
    # get tne number of cliques with size > 2

    return get_n_cliques_gtN(network, 2)

def run_network_modularity(people, network, resolution = 1):
    # network modularity measurement : https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.quality.modularity.html
    # We may want to try different resolution parameters, >1 favors larger communities, <1 favors smaller communities

    # get the communities
    pods, _ = get_pods(network, resolution = resolution)

    # return the modularity
    return nx.algorithms.community.modularity(network, pods, resolution = resolution)

metrics = [
    run_frac_mentees_less_than_requested,
    run_frac_mentors_assigned_mentees,
    run_frac_mentors_with_extra_slots,
    run_frac_mentees_atleast_one_preference,
    run_mean_clique_size,
    run_n_cliques_gt2,
    run_frac_mentees_alternatives,
    run_network_modularity]

#### functions that call the above:
def run_all_metrics(people,network):

    metric_values = [metric(people,network) for metric in metrics]

    return metric_values,[metric.__name__.split('run_')[1] for metric in metrics]

def run_weighted_metrics(people_list, network_list, metrics, combine_metric_method='multiply', minvalue=0.01):
    '''
    metrics is a list of dict's with each dict containing the following keys.
        function : the name of the metric function (e.g., run_frac_mentees_with_a_mentor)
        weight: a numerical weight to give to that metric
        type: the type of weigthing to use.  Options include:
            maximize : higher numbers get preference
            minimize : lower numbers get prefererence
            binary : non-zero numbers are converted to 1 (and given preference)
            binary0: zeros are converted to 1 (and given preference); all other numbers are converted to zero
        normalize : boolean, if True then the metric values are normalized to [0,1].  This step is performed before weighting
        minvalue : float value that sets the minimum value for that metric (optional, = 0.01 by default as defined in the function args).

    combine_metric_method can be either 'multiply' or 'mean', see code for both methods ('multiply is the default')

    Note: if 'weight' is left blank or not included, then a weight of 1 in assumed
    Note: if 'type'is left blank or not included, then no weighting is applied
    Note: in the current version, if we want to "throw out" any runs (i.e, se the combined_metric to zero), 
          we need to set combine_metric_method to 'multiply'
    '''
    def norm_metric(arr):
        return (arr - min(arr))/(max(arr) - min(arr))

    nruns = len(network_list)
    nmetrics = len(metrics)

    metric_values = np.zeros((nmetrics, nruns))
    weighted_metric_values = np.zeros((nmetrics, nruns))
    combined_metric = np.zeros(nruns)
    metric_names = [] #for output plotting

    # it would be nice to do this and the combined metric in one loop, 
    # but I think in order to normalize the metric to [0,1] (an input option), we will need to first get 
    #    values for that metric across all runs
    for i,m in enumerate(metrics):

        # populate this metric for all runs
        metric = m['function']
        metric_names.append(metric.__name__.split('run_')[1])

        for j in range(nruns):
            # run the metric
            metric_values[i,j] = metric(people_list[j], network_list[j])

        # perform the weighting and normalization (if necessary)
        weighted_metric_values[i,:] = metric_values[i,:]

        if ('normalize' in m):
            if (m['normalize']):
                weighted_metric_values[i,:] = norm_metric(metric_values[i,:])

        if ('weight' not in m):
            m['weight'] = 1

        if ('minvalue' not in m):
            m['minvalue'] = minvalue

        if ('type' in m):
            if (m['type'] == 'maximize'):
                weighted_metric_values[i,:] = np.where(weighted_metric_values[i,:] > 0, weighted_metric_values[i,:], m['minvalue'])
                weighted_metric_values[i,:] = m['weight']*weighted_metric_values[i,:]
            elif (m['type'] == 'minimize'):
                weighted_metric_values[i,:] = (1. - weighted_metric_values[i,:])
                weighted_metric_values[i,:] = np.where(weighted_metric_values[i,:] > 0, weighted_metric_values[i,:], m['minvalue'])
                weighted_metric_values[i,:] = m['weight']*weighted_metric_values[i,:]
            elif (m['type'] == 'binary'):
                weighted_metric_values[i,:] = weighted_metric_values[i,:] > 0
                weighted_metric_values[i,:] = m['weight']*weighted_metric_values[i,:].astype(int)
            elif (m['type'] == 'binary0'):
                weighted_metric_values[i,:] = weighted_metric_values[i,:] == 0
                weighted_metric_values[i,:] = m['weight']*weighted_metric_values[i,:].astype(int)


    # calculate the combined metric for each run
    for j in range(nruns):

        if (combine_metric_method == 'mean'):
            # perform a weighted mean
            numerator = 0.
            denominator = 0.
            for i,v in enumerate(weighted_metric_values[:,j]):
                numerator += v
                denominator += metrics[i]['weight']
            combined_metric[j] = numerator/denominator
        else:
            # multiply the metric together and divide by the weights
            numerator = 1.
            denominator = 1.
            for i,v in enumerate(weighted_metric_values[:,j]):
                numerator *= v
                denominator *= metrics[i]['weight']
            combined_metric[j] = numerator/denominator


    return {'raw_metrics':metric_values, 
            'weighted_metrics':weighted_metric_values, 
            'combined_metric':combined_metric, 
            'metric_names':np.array(metric_names)
            }

def create_best_network(
    nruns,
    names_df,
    mentees_df,
    mentors_df,
    metrics,
    nbest = 1,
    combine_metric_method='multiply',
    loud=False,
    seed=None):
    # wrapper to run all the code needed to create a network

    
    ## set the random seed. if None will use default seed in set_seed
    mentor_matching.set_seed(seed)

    network_list = []
    people_list = []
    for i in range(nruns):
        people, network = mentor_matching.generate_network(names_df,mentees_df,mentors_df,loud)
        flag = True
        for constraint in constraints: flag = flag and constraint(people,network)

        ## network passed muster, add it to the list
        if flag:
            network_list.append(network)
            people_list.append(people)
        else: print("A network violated a constraint and was discarded.")

    output = run_weighted_metrics(people_list, network_list, metrics, combine_metric_method)

    output['people_list'] = people_list
    output['network_list'] = network_list

    bestlist = ([])
    sorted = output['combined_metric'].argsort()
    best_index = sorted[::-1][:nbest]
    for ibest in range(nbest):
        if ibest == 0:
            output['best'] = {
            'people':people_list[best_index[ibest]],
            'network':network_list[best_index[ibest]],
            'combined_metric':output['combined_metric'][best_index[ibest]],
            'index':best_index[ibest],
            }
        bestlist.append({'people':people_list[best_index[ibest]],
            'network':network_list[best_index[ibest]],
            'combined_metric':output['combined_metric'][best_index[ibest]],
            'index':best_index[ibest],
            })
    output['bestlist'] = bestlist
            

    return output
