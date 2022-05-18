import pandas as pd

## open the CSV files from Kari
##  the CIERA directory (https://ciera.northwestern.edu/directory/)
directory_export = pd.read_csv('data/directory_export.csv')

directory_names = directory_export['First Name'] + ' ' + directory_export['Last Name']
for i,name in enumerate(directory_names):
    if '  ' in name: 
        directory_names[i] = name.replace('  ',' ')

##  internal list that Kari keeps
ciera_members = pd.read_csv('data/ciera_members.csv')

## make a dataframe that will do the bulk of the merging.
translator = pd.DataFrame({})


translator['Name'] = directory_names
translator['Email'] = directory_export['Email']

## make the final output dataframe
final = pd.DataFrame({})
member_names = ciera_members['Name']
for i,name in enumerate(member_names):
    if '  ' in name: 
        member_names[i] = name.replace('  ',' ')

final['Name'] = member_names
final['Role'] = ciera_members['Current Type']

## set all indices to be the Name for easier cross-matching
final = final.set_index('Name')
ciera_members = ciera_members.set_index('Name')
translator = translator.set_index('Name')
directory_export = directory_export.set_index(directory_names)

## outer join will take the union of the two csv files
final = final.join(translator,how='outer',)

## handle missing final ciera_members for translator entries
missing_ciera_members = final.loc[final.Role.isna(),'Role']
final.loc[missing_ciera_members.index,'Role'] = directory_export.loc[missing_ciera_members.index,'Person Types']
final = final.loc[~final.Role.isna()]

## handle missing translator emails for final entries
##  put NetID as a placeholder, easiest way to find someone's email is to use 
##  `finger <netid>` on quest
final.loc[final.Email.isna(),'Email'] = ciera_members.loc[final[final.Email.isna()].index].NetID


## make ciera_members uniform
final.loc[final.Role.str.contains('Professor'),'Role'] = 'Faculty'
final.loc[final.Role.str.contains('Faculty'),'Role'] = 'Faculty'
final.loc[final.Role.str.contains('Grad'),'Role'] = 'Graduate Student'
final.loc[final.Role.str.contains('Undergraduate'),'Role'] = 'Undergraduate Student'
final.loc[final.Role.str.contains('Postdoc'),'Role'] = 'Postdoc'

print(final.Role.unique())

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