import win32com.client as win32

def generate_automatic_outlook_emails(emails, N=np.inf):

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