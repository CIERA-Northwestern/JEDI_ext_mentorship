import itertools
import networkx as nx

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

def find_cliques_size_k(G, k):
    # following : https://stackoverflow.com/questions/58775867/what-is-the-best-way-to-count-the-cliques-of-size-k-in-an-undirected-graph-using

    i = 0
    for clique in nx.find_cliques(G):
        if len(clique) == k:
            i += 1
        elif len(clique) > k:
            i += len(list(itertools.combinations(clique, k)))
    return i

def run_mean_clique_size(people, network, max_clique_size = 10):
    # get the mean clique size for the network.  Presumably this should be maximized

    mean_clique_size = 0
    denom = 0
    max_clique_size = 10
    G = network.to_undirected()
    for i in range(max_clique_size):
        n = find_cliques_size_k(G, i)
        #print("cliques", i, n)
        mean_clique_size += i*n
        denom += n

    if (denom > 0):
        mean_clique_size /= denom

    return mean_clique_size

def run_n_cliques_gt3(people, network, max_clique_size = 10):
    # get tne number of cliques with size > 3 (could change 3 to any number)

    n_cliques = 0
    G = network.to_undirected()
    for i in range(max_clique_size):
        if (i > 3):
            n_cliques += find_cliques_size_k(G, i)

    return n_cliques

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
        run_n_cliques_gt3]

    metric_values = [metric(people,network) for metric in metrics]

    return metric_values,[metric.__name__.split('run_')[1] for metric in metrics]
