from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
style.use('fivethirtyeight')
# r and k are classes having two dimensional features 
dataset={'k':[[1,2],[2,3],[3,1]],'r':[[6,5],[7,7],[8,6]]} 
new_features=[5,7]

#for i in dataset:
    #for ii in dataset[i]:
#[[plt.scatter(ii[0],ii[1], s=100, color=i)for ii in dataset[i]] for i in dataset]
#plt.scatter(new_features[0],new_features[1])
#plt.show() 
      
#euclidean_distance=sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )
#print(euclidean_distance)

# dat is the testing data (groups)   
def k_nearest_neighbors(data,predict,k=3): 
    if len(data) >=k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances=[]
    for  group in data:
        for features in data[group]:
            euclidean_distance=np.linalg.norm(np.array(features)-np.array(predict)) #using linear algebra inbuilt function it si faster
            # euclidean_distance =np.sqrt(np.sum(( np.array(features) -np.array(predict))**2))  #using normal formaula dynamic but little slower for large dat
            distances.append([euclidean_distance ,group])
    
    votes=[i[1] for i in sorted(distances)[:k]]  #  i[1] is group in distance
    print(votes)
    print(Counter(votes).most_common(1))
    vote_result =Counter(votes).most_common(1)[0][0] #[0][0] first 0 gives  common list second 0 gives you the common first list of array secod (most common group and how many they were)
    print(vote_result)
    
    
    return vote_result 
result=k_nearest_neighbors(dataset , new_features ,k=3)
print(result)






