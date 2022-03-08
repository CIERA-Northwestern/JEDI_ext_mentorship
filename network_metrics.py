import itertools
import networkx as nx
import numpy as np

def run_frac_mentees_with_a_mentor(people,network):
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
    return num/denom

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
    
def run_frac_mentors_overassigned(people,network):
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

def run_frac_any_avoid(people,network):
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
    return num/len(people)
    
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
            if (np.any(person.n_role_mentors - person.has_n_role_mentors) < 0):
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


def run_all_metrics(people,network):

    metrics = [
        run_frac_mentees_with_a_mentor,
        run_frac_mentees_less_than_requested,
        run_frac_mentors_assigned_mentees,
        run_frac_mentors_with_extra_slots,
        run_frac_mentors_overassigned,
        run_frac_mentees_atleast_one_preference,
        run_frac_any_avoid,
        run_mean_clique_size,
        run_n_cliques_gt2,
        run_frac_mentees_alternatives]

    metric_values = [metric(people,network) for metric in metrics]

    return metric_values,[metric.__name__.split('run_')[1] for metric in metrics]
