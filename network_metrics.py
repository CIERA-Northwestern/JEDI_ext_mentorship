

def run_frac_mentees_with_a_mentor(people,network):
    """ Count the fraction of people who requested mentors
        but did not receive as many mentors as requested """
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
            if (person.n_mentors_total - len(person.mentor_matches)):
                num+=1
    ## return the fraction
    return num/denom

def run_frac_mentors_assigned_mentees(people,network):
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
    num = 0
    denom = 0
    for person in people.values():
        ## if this person requested any mentors at all
        if (person.n_mentees_total):
            denom+=1
            ## if they didn't get as many as they offered
            if (person.n_mentors_total - len(person.mentor_matches)):
                num+=1
    ## return the fraction
    return num/denom

def run_frac_mentees_atleast_one_preference(people,network):
    num = 0
    denom = 0
    for person in people.values():
        ## if this person requested any mentors at all
        if (person.n_mentors_total):
            denom+=1
            for mentor in person.mentor_matches:
                ## if they preferenced this mentor
                if (mentor.name in person.mentors_prefr):
                    num+=1
                    break
    ## return the fraction
    return num/denom

def run_frac_any_avoid(people,network):
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

def run_all_metrics(people,network):

    metrics = [
        run_frac_mentees_with_a_mentor,
        run_frac_mentees_less_than_requested,
        run_frac_mentors_assigned_mentees,
        run_frac_mentors_with_extra_slots,
        run_frac_mentees_atleast_one_preference,
        run_frac_any_avoid]

    metric_values = [metric(people,network) for metric in metrics]

    return metric_values,[metric.__name__.split('run_')[1] for metric in metrics]