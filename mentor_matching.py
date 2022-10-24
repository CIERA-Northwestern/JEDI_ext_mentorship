import numpy as np
from operator import attrgetter
import random
import networkx as nx


GLOBAL_max_mentees = 6

def set_seed(seed=None): 
    if seed is None: seed = 'beta'
    random.seed(a=seed)

## these are the roles that we will consider for the network
roles = ['Faculty','Postdoc','Graduate Student','Undergraduate Student']


## define some "constant" dictionaries that help us reformat the data
role_transformer = {
    'Undergraduate student / Post-Bac':'Undergraduate Student',
    'Undergraduate':'Undergraduate Student',
    'Undergraduate student':'Undergraduate Student',
    'Undergraduate Student':'Undergraduate Student',
    'Undergraduate students':'Undergraduate Student',
    'Graduate student':'Graduate Student',
    'Graduate Student':'Graduate Student',
    'Graduate students':'Graduate Student',
    'Graduate Students':'Graduate Student',
    'Postdoc':'Postdoc',
    'Postdocs':'Postdoc',
    'Faculty':'Faculty',
    'Staff':'Staff',
    'Number of mentees':'Number of mentees'
}

## in order to rank order
role_ranks = {
    'Undergraduate Student':0,
    'Graduate Student':1,
    'Postdoc':2,
    'Faculty':3,
    'Staff':None}

## define columns where we expect answers to start/end for mentees/mentors
mentee_answers_start = {
    'Undergraduate Student':1,
    'Graduate Student':7,
    'Postdoc':12,
    'Faculty':16    
}

mentee_answers_end = {
    'Undergraduate Student':mentee_answers_start['Graduate Student'],
    'Graduate Student':mentee_answers_start['Postdoc'],
    'Postdoc':mentee_answers_start['Faculty'],
    'Faculty':100} ## dummy index >> length of answers for slicing

mentor_answers_start = {
    'Undergraduate Student':1,
    'Graduate Student':4,
    'Postdoc':9,
    'Faculty':15    
}

mentor_answers_end = {
    'Undergraduate Student':mentor_answers_start['Graduate Student'],
    'Graduate Student':mentor_answers_start['Postdoc'],
    'Postdoc':mentor_answers_start['Faculty'],
    'Faculty':100} ## dummy index >> length of answers for slicing


## workhorse class for accessing preference data
class Person(object):
    def __lt__(self,other):
        return self.rank < other.rank

    def __repr__(self,only_role=False):
        if not only_role: return f"{self.role}: {self.name}"
        else: return f"{self.role[0]}"#+"$_{"+f"({self.has_n_mentees},{self.has_n_mentors}"+")}$"
    
    def __init__(self,name, role,email=None, raise_error=True):
        self.initials = ''.join([part[0] for part in name.split(' ') if part != ''])
        self.name = name.replace(' ','')
        self.fullName = name
        self.role = role_transformer[role]
        self.email = email
        self.raise_error = raise_error
        self.rank = role_ranks[self.role]
        self.years = 0
        
        self.mentees_prefr = []
        self.mentors_prefr = []
        self.mentees_avoid = []
        self.mentors_avoid = []
        
        self.n_mentees_prefr = 0
        self.n_mentors_prefr = 0
        self.n_mentees_avoid = 0
        self.n_mentors_avoid = 0

        self.n_mentors_prefr_for_sorting = 0
        
        ## how many times does this person
        ##  appear in others' lists
        ##  (potentially problematic info to have <__<)
        self.n_other_ee_p = 0
        self.n_other_or_p = 0
        self.n_other_ee_a = 0
        self.n_other_or_a = 0
        
        self.n_role_mentees = np.zeros(4)
        self.n_role_mentors = np.zeros(4)
        
        self.has_n_role_mentees = np.zeros(4)
        self.has_n_role_mentors = np.zeros(4)

        self.n_mentees_max = 0
        self.n_mentees_total = 0
        self.n_mentors_total = 0
        
        self.mentor_matches = []
        self.mentee_matches = []

        self.has_n_mentees = 0
        self.has_n_mentors = 0
        
        self.mentees_remaining = None
        self.mentors_remaining = None

    def parse_row_role_mentee(self,row):
        #print('mentee - ',self)
        reported_role = self.check_role(row)
        if reported_role is not None: raise NotImplementedError
        
        ## let's get the answers (and the associated questions) from this row
        questions,answers = self.get_own_answers(row,mentee_answers_start,mentee_answers_end)
        
        ## unpack the preferences
        prefr_avoid_answers = answers[-2:]
        
        ## handle inconsistent ordering, just in case.
        if 'NOT' in questions[-2:][0]: avoids,prefrs = answers[-2:]
        else: prefrs,avoids = answers[-2:]
        
        if prefrs != 'nan': self.mentors_prefr = prefrs.replace(' ','').split(';')
        if avoids != 'nan': self.mentors_avoid = avoids.replace(' ','').split(';')

        for question,answer in zip(questions,answers[:-2]):
            mentor_role = question.split('[')[1].split(']')[0]
            mentor_role = mentor_role.split(' mentor')[0].split(' peer')[0]
            mentor_role = role_transformer[mentor_role]
            role_index = ['Undergraduate Student','Graduate Student','Postdoc','Faculty'].index(role_transformer[mentor_role])
        
            self.n_role_mentors[role_index]+= int(eval(answer)) if answer != 'nan' else 0
        
        
    def parse_row_role_mentor(self,row):
        #print('mentor - ',self)
        reported_role = self.check_role(row)
        if reported_role is not None: 
            orig_role = self.role
            self.role = reported_role
        
        ## let's get the answers (and the associated questions) from this row
        questions,answers = self.get_own_answers(row,mentor_answers_start,mentor_answers_end)
        
        ## unpack the preferences
        prefr_avoid_answers = answers[-2:]
        
        ## handle inconsistent ordering, just in case.
        if 'NOT' in questions[-2:][0]: avoids,prefrs = answers[-2:]
        else: prefrs,avoids = answers[-2:]
        
        if prefrs != 'nan': self.mentees_prefr = prefrs.replace(' ','').split(';')
        if avoids != 'nan': self.mentees_avoid = avoids.replace(' ','').split(';')

        if reported_role is not None: self.role = orig_role
        
        ## flag to replace Any with min(global max, sum roles specified)
        ##  for 'Number of mentees' mentor_role
        replace_n_mentees_max = False

        for question,answer in zip(questions,answers[:-2]):
            mentor_role = question.split('[')[1].split(']')[0]

            if answer == '5+': answer = GLOBAL_max_mentees
            elif answer == 'nan': answer = 0
            elif answer == 'Any': 
                answer = GLOBAL_max_mentees
                ## flag to come back and make sure to replace this with the sum
                ##  of the mentees in each role if that's smaller
                if mentor_role == 'Number of mentees': replace_n_mentees_max = True
            else: answer = int(eval(answer)) 
            
            if mentor_role == 'Number of mentees': 
                self.n_mentees_max += min(GLOBAL_max_mentees,answer)
                continue
            role_index = ['Undergraduate Student','Graduate Student','Postdoc','Faculty'].index(role_transformer[mentor_role])
        
            self.n_role_mentees[role_index]+= answer
        
        ## if someone said they'd take any number of mentees, replace that max number with the sum of
        ##  the roles they specified or the global max, whichever is smaller
        if replace_n_mentees_max: 
            self.n_mentees_max = min([GLOBAL_max_mentees,np.sum(self.n_role_mentees)])
    
    def check_role(self,row):
        reported_role = role_transformer[row['Role']]
        if self.role != reported_role:
            if self.raise_error: raise ValueError(f"{self} incorrectly listed themselves as {reported_role}")
            else: return reported_role
        return None

    def validate_prefr_avoid(self,name_ranks,raise_error=True):
        

        for prefix in ['mentee','mentor']:
            for suffix in ['prefr','avoid']:
                llist = getattr(self,f'{prefix}s_{suffix}')
                popped = 0
                for i in range(len(llist)):
                    name = llist[i-popped]
                    
                    ## someone had a trailing semi-colon I think
                    if len(name.replace(' ','')) == 0:
                        llist.pop(i-popped)
                        popped+=1
                        continue
                        
                    ## first check that everyone *exists*
                    if name not in name_ranks.keys(): 
                        if self.raise_error: raise NameError(
                            f"{name} does not appear in full participant list" +
                            f" but appears in {self}'s {prefix} {suffix} list.'" +
                            " i.e. a preference was named who did not fill out a form (or their name is misspelled).")
                    
                    ## throw out hierarchically disallowed matches (i.e. a postdoc mentoring a faculty member)
                    ##  however, let's keep such inconsistent info in case someone puts it in avoid to ensure
                    ##  they're not matched with that person in the reverse relationship (just in case)
                    elif (suffix != 'avoid' and 
                        prefix == 'mentor' and 
                        name_ranks[name] < role_ranks[self.role]): 
                        if self.raise_error: raise NameError(
                            "{name} is of rank {rank:d}".format(
                                name=name,
                                rank=name_ranks[name]) +
                            f"which is an invalid preference for {self} {prefix}")

                        ## remove this invalid preference from their list
                        llist.pop(i-popped)
                        popped+=1
                        continue
                        
                    elif ( suffix != 'avoid' and 
                        prefix == 'mentee' and 
                        name_ranks[name] > role_ranks[self.role]): 
                        if self.raise_error: raise NameError(
                            "{name} is of rank {rank:d}".format(
                                name=name,
                                rank=name_ranks[name]) +
                            f"which is an invalid preference for {self} {prefix}")
                            
                        ## remove this invalid preference from their list
                        llist.pop(i-popped)
                        popped+=1
                        continue
                        
    def count_own_appearances(self,people):

        for this_person in people.values():
            self.n_other_ee_p += self.name in this_person.mentees_prefr
            self.n_other_ee_a += self.name in this_person.mentees_avoid
            
            self.n_other_or_p += self.name in this_person.mentors_prefr
            self.n_other_or_a += self.name in this_person.mentors_avoid
                        
    
    def get_own_answers(
        self,
        row,
        role_answers_start_dict,
        role_answers_end_dict):
        
        ## find where in the row this person's answers start
        answers = np.array(row.values[4:],dtype=str)
        questions = np.array(row.keys()[4:],dtype=str)

        if '[Years]' not in questions[0]: 
            raise IOError(
                "Data is not in correct format. "+
                "First question should be years at institution.")
        
        self.years = eval(answers[0]) if answers[0]!='5+' else 9

        ## find the first column (that isn't years) that is False (i.e. !='nan')
        answers_start = np.argmin(answers[1:]=='nan')+1 ## skip first element, is years for all roles
        role_answers_start = role_answers_start_dict[self.role]
        role_answers_end = role_answers_end_dict[self.role]
        
        ## this can happen if they skip the first (few) question(s)
        if answers_start != role_answers_start:
            ## they should not be able to answer questions before their
            ##  role's section of the results spreadsheet
            if answers_start < (role_answers_start): 
                raise ValueError(
                    f"Something went wrong, answers start at {answers_start}" +
                    f"when they should start at {role_answers_start}")
            elif answers_start > (role_answers_start):
                raise IOError(
                    "Data is not in correct format. "+
                    "Answers start at a later column than expected.")
                ## let's fill in 0 (since the questions that were skipped
                ##  are for how many mentees you'd want) <--- not sure this is true anymore
                answers[role_answers_start:answers_start] = '0'
        return questions[role_answers_start:role_answers_end],answers[role_answers_start:role_answers_end]
    
    def print_preferences(self,show_appearances=False):
        print(self)
        print('nmentors:',f'{self.n_mentors_total:0.0f}',self.n_role_mentors)
        print('nmentees:',f'{self.n_mentees_total:0.0f}',self.n_role_mentees,f'({self.n_mentees_max})')
        
        print('------')
        print('avoid  mentees:',self.mentees_avoid)
        print('avoid  mentors:',self.mentors_avoid)
        print('------')
        print('prefer mentees:',self.mentees_prefr)
        print('prefer mentors:',self.mentors_prefr)
        if show_appearances:
            print('------')
            print('other avoid  mentor/mentee:',f'{self.n_other_or_a} {self.n_other_ee_a}')
            print('other prefer mentor/mentee:',f'{self.n_other_or_p} {self.n_other_ee_p}')
        print('------')
        print()
    
    def print_status(self):
        print(self)
        print('nmentors:',f'{self.n_mentors_total:0.0f}',self.n_role_mentors)
        print('matched :',f'{self.has_n_mentors:0.0f}',self.has_n_role_mentors)
        print('------')
        print('nmentees:',f'{self.n_mentees_total:0.0f}',self.n_role_mentees,f'({self.n_mentees_max})')
        print('matched :',f'{self.has_n_mentees:0.0f}',self.has_n_role_mentees)
        print()
        

    def check_mentor_available(self, mentee):
        # check if a given mentor can accept a new mentee in this role
        # this is only relevant for mentors
        # check that the mentor can accept another mentee 
        #print('check', self.name, self.n_mentees_max,  self.n_role_mentees[mentee.rank], self.n_mentees_max - len(self.mentee_matches), self.n_role_mentees[mentee.rank] - self.has_n_role_mentees[mentee.rank])

        # number is less than global max
        check_available = (len(self.mentee_matches) < self.n_mentees_max)
        
        # number in specific role is less than max in that role
        check_available_role = (self.has_n_role_mentees[int(np.floor(mentee.rank))] < self.n_role_mentees[int(np.floor(mentee.rank))])

        return (check_available and check_available_role)

    def check_mentor_needed(self, mentor):
        # check if a given mentee needs a mentor in this role
        # this is only relevant for mentees
        # check that the mentee needs a mentor from this role
        
        check_needed_role = (
            # number in specific role is less than needed in that role
            self.has_n_role_mentors[int(mentor.rank)] < self.n_role_mentors[int(mentor.rank)])
        if self.role == mentor.role:
            check_needed_role = (
            self.has_n_role_mentors[int(mentor.rank)] < self.n_role_mentors[int(mentor.rank)]
            # mentor "outranks" the mentee by more than 1 year (for peer mentoring)
            and mentor.years - self.years > 1)

        return check_needed_role

    def check_compatability(self,other, loud = False):
    ## assuming that self is a mentee and other is a mentor!
        check_avoid = (not (self.name in other.mentees_avoid or self.name in other.mentors_avoid or
        other.name in self.mentees_avoid or other.name in self.mentors_avoid) and self is not other)
        if loud and not check_avoid: print('check neither on avoid lists', check_avoid, '\n mentee name', self.name, '\n mentor avoid list:', other.mentees_avoid,'\n mentor name', other.name, '\n mentee avoid list', self.mentors_avoid)
        #now checking that there is not relation yet, assuming other is always the mentor)
        check_relation = (other not in self.mentor_matches and self not in other.mentee_matches)
        if loud and not check_relation: print('check that this exact relation does not exist yet', check_relation, '\n mentee name', self.name, '\n mentor\'s matches:', other.mentee_matches,'\n mentor name', other.name, '\n mentee\'s matches:', self.mentor_matches)

        return (check_avoid and check_relation)
                
    
## let's read in the data to some intelligible format and get rid of all those nans
##  this will also let us validate the data and flag any errors
def reduce_full_tables(names_df, mentees_df,mentors_df):
    # read in the names file

    # clean up the accents
    names_df['Name'] = names_df['Name'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8').replace(' ','')
    names_df.sort_values(['Name','Role'],inplace=True,ignore_index=True)

    people = {}
    name_ranks = {}
    for index,row in names_df.iterrows():
        ## save a numerical "rank" based on the rolw for later
        ## only inlcude members with ranks that match the roles list above
        name = row['Name'].replace(' ','')
        try:
            rank = roles[::-1].index('Faculty')
            ## initialize the person instances
            people[name] = Person(row['Name'], row['Role'], row['Email'])
            name_ranks[name] = rank

        except:
            raise

    
    ## let's loop through the mentees and read their preferences
    keys = mentees_df.keys()
    for index,this_row in mentees_df.iterrows():
        ## check if the name is in our people dict
        name = this_row['Name'].replace(' ','')
        in_people = name in people
        if (not in_people):
            ## see if they included middle names, and only take first and last
            foo = this_row['Name'].split()
            name = foo[0] + foo[-1]
        try: 
            this_person = people[name]
        except KeyError: 
            raise NameError(
                "{name} does not appear in full participant list.".format(name=this_row['Name'])
                +" i.e. they are not in CIERA but they filled out the form (or their name is misspelled)")

        ## parse preferences from raw row data
        this_person.parse_row_role_mentee(this_row)
        
    ## let's loop through the mentors and read their preferences
    keys = mentors_df.keys()
    for index in mentors_df.index:
        this_row = mentors_df.loc[index]
        try: this_person = people[this_row['Name'].replace(' ','')]
        except KeyError: raise KeyError(
            "{name} does not appear in full participant list.".format(name=this_row['Name']))

        ## parse preferences from raw row data
        this_person.parse_row_role_mentor(this_row)
    
    for this_person in people.values():

        ## remove the null "no"/"No" value if it appears
        for llist in [
            this_person.mentees_prefr,
            this_person.mentees_avoid,
            this_person.mentors_prefr,
            this_person.mentors_avoid]:
            popped = 0
            for i in range(len(llist)):
                if llist[i] in ['no','No']:
                    llist.pop(i-popped)
                    popped+=1

        ## count preferences so we can sort by them in assignment step
        this_person.validate_prefr_avoid(name_ranks)
        this_person.n_mentees_prefr += len(this_person.mentees_prefr)
        this_person.n_mentees_avoid += len(this_person.mentees_avoid)
        this_person.n_mentors_prefr += len(this_person.mentors_prefr)
        this_person.n_mentors_avoid += len(this_person.mentors_avoid)

        ## people who don't have any preferences should
        ##  go last if we're sorting by increasing number 
        ##  of preferences (to match the pickiest people first)
        this_person.n_mentors_prefr_for_sorting = (
            this_person.n_mentors_prefr if 
            this_person.n_mentors_prefr > 0 else
            100)
        
        this_person.n_mentees_total += min(np.sum(this_person.n_role_mentees),this_person.n_mentees_max)
        this_person.n_mentors_total += np.sum(this_person.n_role_mentors)
        
        ## loop through dictionary and count how many times this person
        ##  appears in other people's preferences
        ##  for sorting purposes
        this_person.count_own_appearances(people)
        
        this_person.mentees_remaining = this_person.n_mentees_total
        this_person.mentors_remaining = this_person.n_mentors_total

        ## add a fractional part to separate out years at institution
        ##  anybody w/ 5+ has 9 -> 0.9
        this_person.rank+=this_person.years/10
        
    return people

def generate_network(names_df,mentees_df,mentors_df,loud=True):
    people = reduce_full_tables(names_df,mentees_df,mentors_df)
    network = nx.MultiDiGraph()
    max_rounds = np.max([value.n_mentors_total for value in people.values()])
    ## add an extra 5 rounds just for good luck in case we have a couple
    ##  of rounds where matching failed and we want to try again
    max_rounds += 5
    ## do all direct matching first before going into the rounds
    direct_matching(people,network,loud)
    mentors,mentees = matching_round(people,network,0,loud)
    ## index each of the matching rounds. 
    ##  Only those mentees with Nmentor <= round_index
    ##  are considered eligible for matching. 
    ##  (this can come into play if someone had many
    ##  direct matches or if we eventually match a mentee with
    ##  multiple mentors in a single round; e.g. to create a pod).
    ##  NOTE: ^ to be clear, this is not implemented, just an idea
    for round_index in range(1,int(max_rounds)):
        ## do a matching round
        mentors,mentees = matching_round(people,network,round_index,loud)
        ## count the remaining mentors and mentees
        ##  if we ran out of one or the other then let's give up
        if len(mentors) == 0 or len(mentees) == 0: break
        
    return people,network
    
def direct_matching(people,network,loud=True):
    ## sort such that people with the fewest mentors requested get 
    ##  matched first (ties broken by rank and those ties broken by
    ##  the number of people preferenced) in the unlikely event 
    ##  that a mentor's availability is filled up.
    mentees = sorted([
        value for value in people.values() if (
        value.mentors_remaining > 0 and  ## only match those who need matches
        value.n_mentors_prefr > 0)], ## only keep mentees that actually have prefered mentors
        key=attrgetter("n_mentors_total","rank","n_mentors_prefr"),
        reverse=False)
    for person in mentees:
        for other_name in person.mentors_prefr:
            if (other_name in people):
                other:Person = people[other_name]
                ## check if they both prefer each other as mentee/mentor (either way)
                if (person.name in other.mentees_prefr):
                    ## double check for compatibility and availability
                    if (person.check_compatability(other, loud=loud) and 
                        other.check_mentor_available(person) and 
                        person.check_mentor_needed(other)):
                      
                        add_relationship(network,other,person,loud=loud)
                
                
def matching_round(people,network,round_index=0,loud=True):
    ## make a list of people who want at least 1 additional mentor, 
    ##  sorted s.t. people who want the fewest mentors are first, 
    ##  with ties broken by number of people prefered (so that people
    ##  with preferences have a better chance of getting those filled)
    ##  and then matching people randomly in order of rank
    ##  If a grad student specifically asks for a postdoc they should
    ##  be matched before an undergrad is randomly assigned IMO.
    mentees = sorted([
        value for value in people.values() if 
        value.mentors_remaining > 0 and 
        value.has_n_mentors <= (round_index)],
        key=attrgetter("n_mentors_total","n_mentors_prefr_for_sorting","rank"),
        reverse=False)

    ## make a list of people who are (still) willing to mentor
    mentors = sorted([
        value for value in people.values() if (value.mentees_remaining)>0],
        key=attrgetter("n_mentees_total"),
        reverse=False)
    mentors = sorted(mentors,key=attrgetter("rank"),reverse=True)

    ## attempt to match each remaining mentee with a mentor
    for mentee in mentees: find_mentor(network,mentee,mentors,loud)

    if loud: print('Mentors remaining:',len(mentors),'Mentees remaining:',len(mentees))
    return mentors,mentees

def find_mentor(network,mentee:Person,mentors,loud):
    mentors_acceptable = ([])
    mentors_alternative = ([])
    mentors_preferred = ([])
    mentors_prefer_mentee = ([])
    for mentor in mentors:
        ##remove mentors to avoid
        ## also remove mentors that want to avoid this mentee
        ## and check that the mentor still has available spots for a mentee in this role
        if (mentee.check_compatability(mentor, loud=loud) and mentor.check_mentor_available(mentee)):
            ## check that this mentee still needs a mentor of that role
            ## (included a larger than 2 year difference for peer mentors)
            if  (mentee.check_mentor_needed(mentor)):
                mentors_acceptable.append(mentor)
                ## check for preferred mentors by this mentee
                if mentor.name in mentee.mentors_prefr:
                    mentors_preferred.append(mentor)
                ## check for mentors that prefer this mentee
                if mentee.name in mentor.mentees_prefr:
                    mentors_prefer_mentee.append(mentor)
            ## provide alternative option:
            ## includes any available mentor than is compatible and more senior (also peers >= 1 yr apart)
            elif (mentor.rank > mentee.rank):
                mentors_alternative.append(mentor)
    if len(mentors_preferred):
        ## there is a preferred mentor still available
        prosp_mentor = random.choice(mentors_preferred)
    elif len(mentors_prefer_mentee):
        ## there is a mentor that prefers this mentee available
        prosp_mentor = random.choice(mentors_prefer_mentee)
    elif len(mentors_acceptable) == 0:
        if len(mentors_alternative):
            random.shuffle(mentors_alternative) ## shuffle occurs inplace 
            sorted_alt_mentors = sorted(mentors_alternative,key=attrgetter("rank"))
            prosp_mentor = sorted_alt_mentors[0]
            if loud: print ('Mentee ', mentee, ' cannot be matched to a mentor of desired role, but will get another suggestion:', prosp_mentor)
        else:
            if loud: print ('Mentee ', mentee, ' cannot be matched to a mentor any more that satisfies all mentee+mentors requirements!')
                ##TODO: need to decide how to handle these cases?
            return
    else:
        ## pick one from the general list with removed avoid mentors
        prosp_mentor = random.choice(mentors_acceptable)
    add_relationship(network,prosp_mentor,mentee)
            
def add_relationship(network,mentor:Person,mentee:Person,loud:bool=False):
    ## update the mentee's status
    mentee.mentor_matches.append(mentor)
    mentee.has_n_role_mentors[int(mentor.rank)] += 1
    mentee.has_n_mentors += 1
    mentee.mentors_remaining -= 1

    ## update the mentor's status
    mentor.mentee_matches.append(mentee)
    mentor.has_n_role_mentees[int(mentee.rank)] += 1
    mentor.has_n_mentees += 1
    mentor.mentees_remaining -=1

    if loud: 
        print(f"matched mentee: {mentee} with mentor: {mentor}")
        mentee.print_status()
        mentor.print_status()

    ## add edge to the network
    return network.add_edge(mentor,mentee)

