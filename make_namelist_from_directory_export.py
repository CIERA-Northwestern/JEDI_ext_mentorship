import pandas as pd

## open the CSV files from Kari
##  the CIERA directory (https://ciera.northwestern.edu/directory/)
directory_export = pd.read_csv('data/directory_export.csv')
directory_export = directory_export.set_index(directory_export['First Name'] + ' ' + directory_export['Last Name'])

##  internal list that Kari keeps
roles = pd.read_csv('data/ciera_members.csv')

## make a dataframe that will do the bulk of the merging.
translator = pd.DataFrame({})
translator['Name'] = directory_export['First Name'] + ' ' + directory_export['Last Name']
translator['Email'] = directory_export['Email']

## make the final output dataframe
final = pd.DataFrame({})
final['Name'] = roles['Name']
final['Role'] = roles['Current Type']

## set all indices to be the Name for easier cross-matching
final = final.set_index('Name')
roles = roles.set_index('Name')
translator = translator.set_index('Name')

## outer join will take the union of the two csv files
final = final.join(translator,how='outer',)

## handle missing final roles for translator entries
missing_roles = final.loc[final.Role.isna(),'Role']
final.loc[missing_roles.index,'Role'] = directory_export.loc[missing_roles.index,'Person Types']
final = final.loc[~final.Role.isna()]

## handle missing translator emails for final entries
##  put NetID as a placeholder, easiest way to find someone's email is to use 
##  `finger <netid>` on quest
final.loc[final.Email.isna(),'Email'] = roles.loc[final[final.Email.isna()].index].NetID

if __name__ == '__main__':
    ## print any entries that are missing values
    print('----- people missing info -----')
    print(final.loc[final.isna().any(axis=1)])
    final = final.dropna()

    ## confirm they've been eliminated
    print('----- after removal -----')
    print(final.loc[final.isna().any(axis=1)])

    print("saving csv to: data/namelist.csv")
    final.to_csv('data/namelist.csv')