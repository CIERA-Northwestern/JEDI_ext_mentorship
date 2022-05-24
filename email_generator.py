import numpy as np

from network_viz import get_positions

def generate_email_list(network, emailSubject = None, emailText = None):
    pods,pos_dicts,missing_edgess,anti_nodess = get_positions(network)

    subject = emailSubject
    if (emailSubject is None):
        subject = "CIERA Mentorship Network match notification"

    output = []
    for pod in pods:
        edges = list(pod.edges())
        for edge in edges:
            mentor = edge[0]
            mentee = edge[1]
            email = dict()
            email["to"] = mentor.email + "; " + mentee.email
            email["subject"] = subject
            text = emailText
            if (emailText is None):
                name1 = mentor.fullName.split()[0]
                name2 = mentee.fullName.split()[0]
                text = f"Dear {name1} and {name2},<br/><br/>You have been matched as a mentor ({name1}) - mentee ({name2}) pair in the CIERA Mentorship Network.  Please use this email as a starting point for your new mentorship relationship.<br/><br/>We suggest reaching out and planning a first meeting now; it doesn't have to take place very soon, it just needs to be scheduled.  For the first meeting try to get to know each other a bit and find out whether there is a preferred focus area for this mentoring relationship.<br/><br/>You can find useful information on mentorship on the <a href='https://sites.northwestern.edu/cieraguide/mentorship/'>CIERA Guide here.</a><br/><br/>We are aware that availability and priorities may change.  If for any reason you feel unable to contribute to this mentoring relationship at this time, please reply-all to this email to let us know.<br/><br/>Sincerely,<br/>The Mentorship Action Team<br/>(A-Z by first name) Aaron Geller, Adam Miller, Alex Gurvich, Nick Kaaz, Sam Imperato, Tjitske Starkenburg, Zoheyr Doctor"
            email["text"] = text

            output.append(email)

    return output



def generate_automatic_outlook_emails(emails, N=np.inf):

    import win32com.client as win32
    outlook = win32.Dispatch('outlook.application')

    def generate_email(email, send=False):
        mail = outlook.CreateItem(0)
        mail.To = email['to']
        mail.Subject = email['subject']
        mail.HTMLBody  = email['text']
        if (send):
            ## this will automatically send the email... seems a bit dangerous, so not hooked in 
            mail.send
        else:
            mail.Display(False) #set to false so that it doesn't stop the script

    for i,email in enumerate(emails):
        generate_email(email)
        if (i >= N-1):
            break