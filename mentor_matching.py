import numpy as np
from operator import attrgetter
import random
import networkx as nx

## define some "constant" dictionaries that help us reformat the data
role_transformer = {
    'Undergraduate student':'Undergrads',
    'Undergraduate students':'Undergrads',
    'Graduate student':'GradStudents',
    'Graduate students':'GradStudents',
    'Postdoc':'Postdocs',
    'Postdocs':'Postdocs',
    'Faculty':'Faculty',
    'Number of mentees':'Number of mentees'
}

## in order to rank order
role_ranks = {
    'Undergrads':0,
    'GradStudents':1,
    'Postdocs':2,
    'Faculty':3}

## define columns where we expect answers to start/end for mentees/mentors
mentee_answers_start = {
    'Undergrads':0,
    'GradStudents':6,
    'Postdocs':11,
    'Faculty':15    
}

mentee_answers_end = {
    'Undergrads':mentee_answers_start['GradStudents'],
    'GradStudents':mentee_answers_start['Postdocs'],
    'Postdocs':mentee_answers_start['Faculty'],
    'Faculty':100} ## dummy index >> length of answers for slicing

mentor_answers_start = {
    'Undergrads':0,
    'GradStudents':3,
    'Postdocs':8,
    'Faculty':14    
}

mentor_answers_end = {
    'Undergrads':mentor_answers_start['GradStudents'],
    'GradStudents':mentor_answers_start['Postdocs'],
    'Postdocs':mentor_answers_start['Faculty'],
    'Faculty':100} ## dummy index >> length of answers for slicing


## workhorse class for accessing preference data
class Person(object):
    def __repr__(self):
        return f"{self.role}: {self.name}"
    
    def __init__(self,name,role,raise_error=False):
        self.name = name.replace(' ','')
        self.role = role
        self.raise_error = raise_error
        self.rank = role_ranks[self.role]
        
        self.mentees_prefr = []
        self.mentors_prefr = []
        self.mentees_avoid = []
        self.mentors_avoid = []
        
        ## I may regret this... trying to keep 
        ##  variable names as short as possible
        ##  length of above lists so we can use attrgetter
        ##  to sort
        self.n_ee_p = 0
        self.n_or_p = 0
        self.n_ee_a = 0
        self.n_or_a = 0
        
        ## how many times does this person
        ##  appear in others' lists
        ##  (potentially problematic info to have <__<)
        self.n_other_ee_p = 0
        self.n_other_or_p = 0
        self.n_other_ee_a = 0
        self.n_other_or_a = 0
        
        self.n_role_mentees = np.zeros(4)
        self.n_role_mentors = np.zeros(4)
        
        self.n_mentees_max = 0
        self.n_mentees_total = 0
        self.n_mentors_total = 0
        
        self.mentor_matches = []
        self.mentee_matches = []
        
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
            role_index = ['Undergrads','GradStudents','Postdocs','Faculty'].index(mentor_role)
        
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
        
        for question,answer in zip(questions,answers[:-2]):
            if answer == '5+': answer = 10
            elif answer == 'nan': answer = 0
            else: answer = int(eval(answer)) 
            
            
            if self.role != 'Undergrads': mentor_role = question.split('[')[1].split(']')[0]
            else: 
                self.n_role_mentees[0] += answer
                mentor_role = 'Number of mentees'
            mentor_role = role_transformer[mentor_role]
            if mentor_role == 'Number of mentees': 
                self.n_mentees_max += answer
                continue
            role_index = ['Undergrads','GradStudents','Postdocs','Faculty'].index(mentor_role)
        
            self.n_role_mentees[role_index]+= answer
        
    
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
        answers_start = np.argmin(answers=='nan')
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
                
            ## let's fill in 0 (since the questions that were skipped
            ##  are for how many mentees you'd want)
            answers[role_answers_start:answers_start] = '0'
        
        return row.keys()[4+role_answers_start:4+role_answers_end],answers[role_answers_start:role_answers_end]
    
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
        
    def check_compatability(self,other):
        return (not (self in other.mentees_avoid or self in other.mentors_avoid or
        other in self.mentees_avoid or other in self.mentors_avoid) and self is not other)
                
    
## let's read in the data to some intelligible format and get rid of all those nans
##  this will also let us validate the data and flag any errors
def reduce_full_tables(names_df,mentees_df,mentors_df):
    
    ## let's first separate the names into their respective roles and make a look-up table
    roles = ['Faculty','Postdocs','GradStudents','Undergrads']
    role_dict = {} ## name -> role
    for role in roles:
        this_names = names_df[role].dropna().values
        role_dict.update(zip(this_names,np.repeat(role,len(this_names))))

    name_ranks = {}
    for i,key in enumerate(roles[::-1]):
        values = names_df[key].dropna().values
        for name in values:
            name_ranks[name.replace(' ','')] = i
    
    ## then let's initialize the person instances
    people = dict([(name.replace(' ',''),Person(name,role)) for name,role in role_dict.items()])
    
    
    ## let's loop through the mentees and read their preferences
    keys = mentees_df.keys()
    for index in mentees_df.index:
        this_row = mentees_df.loc[index]
        try: this_person = people[this_row['Name'].replace(' ','')]
        except KeyError: raise NameError(
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
        ## count preferences so we can sort by them in assignment step
        this_person.validate_prefr_avoid(name_ranks)
        this_person.n_ee_p += len(this_person.mentees_prefr)
        this_person.n_ee_a += len(this_person.mentees_avoid)
        this_person.n_or_p += len(this_person.mentors_prefr)
        this_person.n_or_a += len(this_person.mentors_avoid)
        
        this_person.n_mentees_total += np.sum(this_person.n_role_mentees)
        this_person.n_mentors_total += np.sum(this_person.n_role_mentors)
        
        ## loop through dictionary and count how many times this person
        ##  appears in other people's preferences
        ##  for sorting purposes
        this_person.count_own_appearances(people)
        
        this_person.mentees_remaining = this_person.n_mentees_total
        this_person.mentors_remaining = this_person.n_mentors_total
        
    return people

def generate_network(names_df,mentees_df,mentors_df,loud=False):
    people = reduce_full_tables(names_df,mentees_df,mentors_df)
    network = nx.MultiDiGraph()
    mentors,mentees = matching_round(people,network,loud)
    nmentors,nmentees = len(mentors),len(mentees)
    while True:
        mentors,mentees = matching_round(people,network,loud)
        this_nmentors,this_nmentees = len(mentors),len(mentees)
        if nmentors == this_nmentors and nmentees == this_nmentees: break
        nmentors = this_nmentors
        nmentees = this_nmentees
        
    return people,network
                
def matching_round(people,network,loud=True):
    ## make a list of people who want at least 1 mentor, sorted s.t. people who
    ##  are most junior first with ties broken by how many mentors they want
    mentees = sorted([
        value for value in people.values() if (value.n_mentors_total-len(value.mentor_matches))>0],
        key=attrgetter("n_mentors_total"),
        reverse=False)
    mentees = sorted(mentees,key=attrgetter("rank"),reverse=False)

    ## make a list of people who are willing to mentor
    mentors = sorted([
        value for value in people.values() if (value.n_mentees_total-len(value.mentee_matches))>0],
        key=attrgetter("n_mentees_total"),
        reverse=False)
    mentors = sorted(mentors,key=attrgetter("rank"),reverse=True)

    ## attempt to match each remaining mentee with a mentor
    for mentee in mentees: find_mentor(network,mentee,mentors)

    if loud: print('Mentors remaining:',len(mentors),'Mentees remaining:',len(mentees))
    return mentors,mentees

def find_mentor(network,mentee:Person,mentors):
    keep_going = True
    while keep_going:
        prosp_mentor = random.choice(mentors)
        if mentee.check_compatability(prosp_mentor):
            #if prosp_mentor.role == mentee.role:
                ## handle peer mentor case
                #if prosp_mentor.check_mentor_compatability(mentee):
                    #add_relationship(network,mentee,prosp_mentor)
                #else: continue
            add_relationship(network,prosp_mentor,mentee)
            keep_going = False
            
def add_relationship(network,mentor,mentee):
    mentee.mentor_matches.append(mentor)
    mentor.mentee_matches.append(mentee)
    network.add_edge(mentor,mentee)