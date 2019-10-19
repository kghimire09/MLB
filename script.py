import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from svm_visualization import draw_boundary
from players import aaron_judge, jose_altuve, david_ortiz
fig, ax = plt.subplots()
aaron_judge['type']=aaron_judge['type'].map({'S':1,'B':0})
aaron_judge=aaron_judge.dropna(subset=  ['type','plate_x','plate_z'])
plt.scatter(x=aaron_judge.plate_x,y=aaron_judge.plate_z,     c=aaron_judge.type, cmap=plt.cm.coolwarm, alpha=0.5)
training_set, validation_set=         train_test_split(aaron_judge, random_state=1)
classifier=SVC(kernel='rbf', gamma=100, C=100)
classifier.fit(training_set[['plate_x','plate_z']],   training_set['type'])
draw_boundary(ax, classifier)
accuracy= classifier.score(training_set[['plate_x','plate_z']], training_set['type'])
ax.set_ylim(-2,6)
ax.set_xlim(-3,3)
plt.show()


