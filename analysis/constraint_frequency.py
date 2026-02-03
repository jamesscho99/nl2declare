import pandas as pd

# import ./data/data_collection.csv
data = pd.read_csv('../data/data_collection.csv')
print(data)

# add a column from "Declare Constraint" by extracting after " and before (", e.g., "AtLeastOne(AttendFieldTrip)" -> AtLeastOne
data['Constraint'] = data['"Declare Constraint"'].str.extract(r'([A-Za-z]+)')

# check if any Constraint value is missing or empty
print(data['Constraint'].isnull().sum())

# visualize the frequency of each constraint using seaborn by sorting the frequency in descending order
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_theme(style="whitegrid")
ax = sns.countplot(y='Constraint', data=data, order = data['Constraint'].value_counts().index)
# set size of the plot
fig = plt.gcf()
fig.set_size_inches(15, 6)
# set font size
plt.xlabel('', fontsize=10)
plt.ylabel('', fontsize=10)
# change label font
plt.xticks(fontname='Times New Roman', fontsize=13)
plt.yticks(fontname='Times New Roman', fontsize=13)
# use times font
# plt.title('Frequency of Constraints')
# for i in ax.patches:
#     ax.text(i.get_width()+.3, i.get_y()+.38, str(i.get_width()), fontsize=10)
plt.savefig('./Constraint_Frequency.pdf')

# # save i
# plt.show()