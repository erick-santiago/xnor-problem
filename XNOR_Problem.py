# DATA GENERATION
import numpy as np
from matplotlib import pyplot as plt

# set up number of training samples
N = 250

# set up upper and lower bounds on our input probabilities
ub = 20
lb = 0.01

# create our X's and O's datasets
x = np.empty((N,2))
o = np.empty((N,2))

# set up the X-class (first and third quadrants)
for i in range(N):
    quad = np.random.randint(0,2)
    
    if quad == 0:
        newX = np.random.uniform(lb,ub) # top quad
        newY = np.random.uniform(lb,ub)
    else:
        newX = np.random.uniform(-lb,-ub) # bot quad
        newY = np.random.uniform(-lb,-ub)
        
    x[i,0] = newX
    x[i,1] = newY
    
# set up the O-class (second and fourth quadrants)
for i in range(N):
    quad = np.random.randint(0,2)
    
    if quad == 0:
        newX = np.random.uniform(-lb,-ub) # top quad
        newY = np.random.uniform(lb,ub)
    else:
        newX = np.random.uniform(lb,ub) # bot quad
        newY = np.random.uniform(-lb,-ub)
        
    o[i,0] = newX
    o[i,1] = newY
    
#plot our two datasets
plt.scatter(x[:,0], x[:,1], marker='+', c='blue', label='x-class')
plt.scatter(o[:,0], o[:,1], marker='o', c='red', edgecolors='none', label='o-class')
plt.legend()
plt.grid(True)
plt.show()
        

#TRAIN DATASET ESTABLISHMENT 
x_train = np.empty((N*2,2))  #training dataset

nd = 0  #counter for training dataset

#set up the training samples from the x-class and the o-class
#and compile the x and o-classes into the x_train dataset
for i in range(len(o)):
    x_train[nd,0] = o[i,0]
    x_train[nd,1] = o[i,1]
    nd += 1
for i in range(len(x)):
    x_train[nd,0] = x[i,0]
    x_train[nd,1] = x[i,1]
    nd += 1

print(len(x_train))  #sanity check
print(x_train[255])
 
#randomly shuffle the sequence of x_train dataset
np.random.shuffle(x_train)
print(x_train[255])  #sanity check

#set up the labels, [1 0] for each x-class and [0 1] for each o-class
y_x = np.array([[1,0]])
y_o = np.array([[0,1]])

#set up the training labels dataset
y_train = np.empty((N*2,2))  

#binary version of x_train, if value pos store 1, if neg store 0
bxt = np.where(x_train[:,:]<0, 0,1) 
for i in range(len(bxt)):
    #perform XNOR check for ea elmt in range and store 0 if different
    y = np.where(bxt[i,0] != bxt[i,1],0,1)  
    if y == 1:
        y_train[i,0] = y_x[0,0]
        y_train[i,1] = y_x[0,1]
    else:
        y_train[i,0] = y_o[0,0]
        y_train[i,1] = y_o[0,1]
        
print(y_train[255])  #sanity check
print(len(y_train))


# NETWORK SETUP AND TRAINING
from keras.models import Sequential
from keras.layers.core import Dense

model = Sequential()
# dense layer w/ 8 hidden nodes, 2 input nodes and ReLU activation
model.add(Dense(8, input_dim=2, activation='relu'))  
# dense layer w/ 2 output nodes and sigmoid activation
model.add(Dense(2, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, batch_size=4, epochs=15, verbose=2)

# list metrics collected in history
print(history.history.keys())

# Plot accuracy and loss vs epoch
plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper right')
plt.show()


# TEST DATASET ESTABLISHMENT

#number of test samples
P = 150

#set up upper and lower bounds on our input probabilities
ub = 20
lb = 0.01

#create our X's and O's datasets
x = np.empty((P,2))
o = np.empty((P,2))

# set up the X-class (first and third quadrants)
for i in range(P):
    quad = np.random.randint(0,2)
    
    if quad == 0:
        newX = np.random.uniform(lb,ub) # top quad
        newY = np.random.uniform(lb,ub)
    else:
        newX = np.random.uniform(-lb,-ub) # bot quad
        newY = np.random.uniform(-lb,-ub)
        
    x[i,0] = newX
    x[i,1] = newY
    
# set up the O-class (second and fourth quadrants)
for i in range(P):
    quad = np.random.randint(0,2)
    
    if quad == 0:
        newX = np.random.uniform(-lb,-ub) # top quad
        newY = np.random.uniform(lb,ub)
    else:
        newX = np.random.uniform(lb,ub) # bot quad
        newY = np.random.uniform(-lb,-ub)
        
    o[i,0] = newX
    o[i,1] = newY
    
z_test = np.empty((P*2,2))  #testing dataset

nd = 0  #counter for testing dataset

#set up the testing samples from the x-class and the o-class
#and compile the x and o-classes into the z_test dataset
for i in range(len(o)):
    z_test[nd,0] = o[i,0]
    z_test[nd,1] = o[i,1]
    nd += 1
for i in range(len(x)):
    z_test[nd,0] = x[i,0]
    z_test[nd,1] = x[i,1]
    nd += 1

print(len(z_test))  #sanity check
print(z_test[255])

#randomly shuffle the sequence of z_test dataset
np.random.shuffle(z_test)  
print(z_test[255])  #sanity check

#set up the labels, [1 0] for each x-class and [0 1] for each o-class
z_x = np.array([[1,0]])
z_o = np.array([[0,1]])

#set up the testing labels dataset
z_labels = np.empty((P*2,2))  

#binary version of z_test, if value pos store 1, if neg store 0
bxt = np.where(z_test[:,:]<0, 0,1) 
for i in range(len(bxt)): 
    #perform XNOR check for ea elmt in range and store 0 if different
    y = np.where(bxt[i,0] != bxt[i,1],0,1)  
    if y == 1:
        z_labels[i,0] = z_x[0,0]
        z_labels[i,1] = z_x[0,1]
    else:
        z_labels[i,0] = z_o[0,0]
        z_labels[i,1] = z_o[0,1]
        
print(z_labels[255])  #sanity check
print(len(z_labels))


# MODEL ACCURACY

# Evaluate the model on the test data 
print("Evaluate on test data")
results = model.evaluate(z_test, z_labels, batch_size=4)
print("test loss, test acc:", results)

