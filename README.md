# CNN_Implementation
Implementation of custom layer in a Convolutional Neural Network on an image dataset.

Implementation of the Layers
Custom Layer Description:
a. Custom Layer Class: A custom layer class is being created by inheriting from the Keras Layer class. This custom class will define how the layer will be computed during the training process.
 b. InitializingtheFiltersProperty:Theclassconstructorinitializesthefiltersproperty to store the number of filters or weight matrices to be used for the current layer. This filter property will be used in the Custom Layer constructor when being used in the model.
c. Initializing Weights: The build() function is called during the model building step, where the weights are initialized using the add_weight() function. The initializer argument is set to "random_normal" to randomly initialize the weights. Additionally, the trainable parameter is set to True to ensure that the weight is updated during training.
d. Defining the Layer Computation Steps: The call() function of the class is defined to perform the layer computation. The function calculates the feature map for each weight matrix in the layer.
e. Looping Through Samples and Feature Maps: The function loops through all the samples and feature maps to calculate the weighted sum of the receptive field using a 3x3 kernel. The receptive field is the area that is considered when applying the filter to the input.
f. Sparse Tensor: After calculating the weighted sum, a sparse tensor is used to generate a matrix with the feature map value to be set into the output map. The sparse tensor is used to efficiently handle large data sets.
g. Updating the Output Map: The output map is then updated with the feature map pixel value for the current receptive field. This process continues until all the receptive fields have been processed.
h. Reshaping the Output Map: Finally, the output map is reshaped to maintain its shape during training. This ensures that the output from the custom layer can be passed on to the next layer in the model without any shape mismatch errors.
CODE:
# Creating the custom layer class by inheriting from Keras Layer class
class Customlayer(keras.layers.Layer): # constructor to initialize the layer
def __init__(self, filters): super(Customlayer, self).__init__() self.filters = filters # building the layer and create the learnable weights
def build(self, input_shape): self.w = self.add_weight(
shape=(self.filters, input_shape[-1], input_shape[-1]), initializer="random_normal",
trainable=True,
)
# Defining the forward pass of the layer
def call(self, inputs):
# getting the batch size dynamically
num_samples = tf.shape(inputs)[0]
# getting the number of input feature maps previous_maps = tf.shape(inputs)[1] # Initializing output tensors with zeros output_maps = tf.zeros((num_samples, self.filters,
tf.shape(inputs)[2]-2, tf.shape(inputs)[3]-2))
# getting the dimensions of the output tensor
axis0,axis1, axis2,axis3= tf.shape(output_maps)
# iterating over the batch, filters and spatial dimensions of
the output tensor
# calculating the feature map for n in range(axis0):

 for k in range(axis1):
# Setting the from and to row and column indices for getting 3X3 kernel
receptive_x1 = tf.constant(0)
kernel,
receptive_y1 = tf.constant(3) receptive_x2 = tf.constant(0) receptive_y2 = tf.constant(3)
#iterating over the spatial dimensions of the output tensor
for i in range(axis2):
# Re-setting the from and to row indices for getting 3X3
# after being moved to the end of matrix
receptive_x1 = tf.constant(0) receptive_y1 = tf.constant(3) for j in range(axis3): weighted_sum = tf.constant(0.0)
# Computing the weighted sum of the input feature maps using the custom weight matrix
for feature in range(previous_maps):
weighted_sum += tf.math.reduce_sum(tf.matmul(inputs[n,
feature, receptive_x1:receptive_y1, receptive_x2:receptive_y2], receptive_x1:receptive_y1, receptive_x2:receptive_y2]))
self.w[k,
# Creating a sparse tensor with the computed weighted sum and its indices
sparse_tensor = tf.SparseTensor(indices=[[n, k, i, j]], values=[weighted_sum],
dense_shape=[axis0, axis1, axis2, axis3]) # adding the weighted sum to the output tensor output_maps = output_maps +
tf.sparse.to_dense(sparse_tensor)
# increasing row-wise stride
receptive_x1 += 1
receptive_y1 += 1
# increasing column-wise stride receptive_x2 += 1 receptive_y2 += 1
# Reshaping output tensor to match the desired shape
return tf.reshape(output_maps, (axis0, axis1, axis2, axis3))
Custom Pooling Layer Description:
a. The code creates a custom pooling layer class by inheriting from the Keras Layer class. In Keras, layers are the basic building blocks of a deep neural network. The CustomPoolingLayers is a custom implementation of the pooling layer that can be used in place of the default pooling layer provided by Keras.
b. The init function initializes the CustomPoolingLayers by calling the init function of the parent class (i.e., the Keras Layer class).

 c. The call function defines the max pooling computation to be performed on each feature map. In this function, the input data is passed through the pooling layer to produce the output.
d. First, the output size of the pooling operation is initialized to half of the original input size.
e. Then, a tensor of zeros is initialized with the same shape as the expected output shape of the pooling operation. This tensor will be filled with the actual pooled values later.
f. The range of indices with a stride size of 2 is initialized to be traversed to get the 2x2 sub-matrix on which pooling is performed. This is done using the tf.range function.
g. The code checks if the input feature map has odd or even dimensions, ignoring the last column and row if the shape is odd. This is done by checking if the end value of the range of indices is odd or even.
h. The axes values for matrix traversal are set based on the shape of the pooled_maps tensor.
i. For each sample and each feature map, the max pooling operation is performed on the 2x2 sub-matrix by using the tf.reduce_max function.
j. The pooled output for each feature map is saved in a tensor.
k. Thepooledoutputforeachsampleinthedatasetissavedinatensor.
l. Finally, the pooled_maps tensor is reshaped and returned in a tensorflow- compatible form. The output shape is consistent with the expected output shape of the pooling operation.
CODE:
# Creating the custom pooling layer class
class CustomPoolingLayers(keras.layers.Layer): # Initializing the CustomPoolingLayers
def __init__(self):
super(CustomPoolingLayers, self).__init__() # Defining the max pooling computation
def call(self, inputs):
# Calculate the output size of the pooled maps by dividing the
input size by 2.
pool_output_size = tf.cast(tf.shape(inputs)[2]/2, tf.int32)
# Create a tensor of zeros for the pooled maps, with dimensions: # (batch size, number
of feature maps, output size, output size).
pooled_maps = tf.zeros((tf.shape(inputs)[0], tf.shape(inputs) [1],
pool_output_size, pool_output_size))
# Determine the end index of the input maps. If the size is odd, the last row and column will be ignored.
end = tf.shape(inputs)[2]
# Checking for input feature map i.e. if it has odd or even dimensions,
if end%2 != 0:
end = tf.shape(inputs)[2] - 1
# Create a tensor of indices for pooling.
indices = tf.range(0, end, 2)
# Get the shape of the pooled maps tensor along each axis.

axis0,axis1,axis2,axis3 = tf.shape(pooled_maps)
# Initialize a TensorArray to store the pooled maps for each
input in the batch
all_pools = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
for n in range(axis0):
# Initialize a TensorArray to store the pooled maps for each
feature map in the input
n_pool = tf.TensorArray(tf.float32, size=0, dynamic_size=True) for k in range(axis1): # Initialize a TensorArray to store the pooled values for each
2x2 region in the feature map.
pool = tf.TensorArray(tf.float32, size=0, dynamic_size=True) for index1 in indices: for index2 in indices:
# Compute the maximum value in the 2x2 region and add it to the pool TensorArray.
pool = pool.write(pool.size(), tf.reduce_max(inputs[n] [k][index1:index1+2, index2:index2+2]))
# Reshape the pooled values into a 2D array with dimensions (output size, output size),
# and add it to the n_pool TensorArray.
n_pool = n_pool.write(n_pool.size(), tf.reshape(pool.stack(), (pool_output_size, pool_output_size)))
# Stack the pooled maps for each feature map into a 3D array with dimensions
# (number of feature maps, output size, output size), and add it to the all_pools TensorArray
all_pools = all_pools.write(all_pools.size(), tf.stack(n_pool.stack()))
# Assign the final max-pooling output in a tensorflow form
pooled_maps = tf.stack(all_pools.stack())
# Reshape the output for a better shape and size while training
return tf.reshape(pooled_maps, (axis0, axis1, axis2, axis3))

The Complete Neural Network Model Description:
a. The neural network outlined below has three proposed layers, two max pooling layers, and adds the Keras Flatten layer to reduce the inputs from the two dimensions to a single dimension vector. However, due to computational constraints, the weight matrices used for each of the three layers are 5, 3, and 2 respectively, rather than 16, 12, or 8.
b. The system can successfully train the model with an image size of 28x28; however, going beyond that requires more than two hours for one epoch.
c. The function initializes a sequential model using the Sequential class from Keras.
d. Acustomlayerwith5weightmatricesisaddedtogenerate5featuremaps.
e. TheReLUactivationfunctionisaddedtointroducenon-linearitytothemodel.
f. A custom pooling layer is added to perform max pooling on the feature maps
generated by the previous layer.
g. Thisisrepeatedwith3and2featuremaps.
 h. The neural network was tested with 16, 12, and 8 layers, but the system's constrained resources prevented it from completing the training process.
i. Overall, the model architecture consists of multiple custom layers that generate feature maps, ReLU activation functions to introduce non-linearity, custom pooling layers to perform max pooling, and fully connected layers to combine the features. The output layer uses the softmax activation function to predict the class probabilities.
CODE:
# Building the neural network model based on what is asked in the assignment
def Buildmodel():
# Initializing the model using Sequential model class of Keras
model = keras.models.Sequential()
# Adding custom layer with 5 weight matrices model.add(Customlayer(filters=5)) # Adding Relu Activation model.add(layers.Activation(activations.relu)) model.add(CustomPoolingLayers())
# Adding custom layer with 3 weight matrices to generate 3 feature maps
model.add(Customlayer(filters=3)) # Adding Relu Activation
model.add(layers.Activation(activations.relu)) model.add(CustomPoolingLayers())
# Adding custom layer with 2 weight matrices to generate 2 feature maps
model.add(Customlayer(filters=2)) # Adding Relu Activation
model.add(layers.Activation(activations.relu))
# Adding Flatten layer
model.add(layers.Flatten())
# Adding Fully Connected layer with 16 neurons model.add(layers.Dense(units=16, activation="relu"))
# To predict class probabilities, an output layer with a single
neuron and a softmax activation function should be added.
model.add(layers.Dense(units=1, activation="softmax")) return model
Implementation and Results
The stages are defined in the method model run that follows.
a. ThemethodModelrun()isusedtocompile,train,andevaluatetheneuralnetwork model.
b. It takes in three parameters: alpha (learning rate), batch_size (number of training samples used in one iteration), and epoch (number of times the training data is passed through the neural network model).
c. A dictionary run_result is initialized to store the learning rate, batch size, epoch, training accuracy, and test accuracy.
d. The adam parameter is a boolean value which, when set to True, uses the Adam optimizer for training the neural network model. Otherwise, it uses the SGD optimizer.
e. The appropriate optimizer is chosen and compiled with the model using the compile() method. The loss function used for optimization is BinaryCrossentropy, and the accuracy metric is used to evaluate the model.

 f. A directory is initialized and set up to store the run files for TensorBoard, a tool for visualizing training data.
g. The fit() method is used to train the model with the given parameters. The X_train and y_train variables represent the input and target output of the training dataset, respectively. The X_valid and y_valid variables represent the input and target output of the validation dataset, respectively. The batch_size and epoch parameters determine the number of training samples used in one iteration and the number of iterations to train the model, respectively. The tensorboardcallbacks is used to save the training progress logs to the directory set up earlier.
h. The training accuracy is saved by extracting the accuracy data from the history object returned by the fit() method.
i. The evaluate() method is used to evaluate the trained model on the test dataset, represented by the X_test and y_test variables.
j. The training accuracy and test accuracy are formatted as percentages and stored in the run_result dictionary.
k. Finally,themodel_historyobjectandrun_resultdictionaryarereturned.
CODE:
# Compile, train, and assess the model
def Modelrun(alpha, batch_size, epoch, adam=False):
# Creating a dictionary named "run_result" and initializing it. run_result = {"Learning Rate": alpha, "Batch Size": batch_size,
"Epoch": epoch, "Train Accuracy": 0.0, "Test Accuracy": 0.0}
# Checking and setting the optimizers (Adam & SGD) to be used for training
if adam:
optimizer = tf.keras.optimizers.Adam(learning_rate=alpha)
else:
optimizer = tf.keras.optimizers.SGD(learning_rate=alpha)
# Compiling the model with given optimizer
model_nn.compile(optimizer=optimizer, loss='BinaryCrossentropy',
metrics=['accuracy'])
# For tensorboard, we now initialise and set up the directory to store run files
Log_directory = "logs/fit/" + datetime.datetime.now().strftime("%Y%m %d- %H%M%S")
tensorboardcallbacks = tf.keras.callbacks.TensorBoard(Log_directory=Log_directory,
 
 histogram_freq=1)
# Training the model
model_history = model_nn.fit(X_train, y_train, epochs=epoch,
batch_size=batch_size, validation_data=(X_valid, y_valid), callbacks=[tensorboardcallbacks])
# Saving the training accuracy obtained from the History object returned by fit method
print("Saving Training Data Accuracy...")
train_acc = model_history.history['accuracy']
# Evaluating test data using the trained model
print("Evaluating Test Data:")
test_loss, test_acc = model_nn.evaluate(X_test, y_test)
# Save results
run_result["Train Accuracy"] = f'{round(np.mean(train_acc) * 100, 4)
}%'
run_result["Test Accuracy"] = f'{round(test_acc * 100, 4)}%'
return model_history, run_result
With the default parameter settings (batch size None and Learning Rate 0.01) of the Stochastic Gradient Descent optimizer offered by Keras Optimizer class, the model generated using the build model method was trained with all data samples using the model run method as follows:
# The results of the run are being saved to test the implementation of the layer and the model.
histories.append(model_history) results.append(run_result)
print("For the default settings of Stochastic Gradient Descent Optimizer the designed model has following accuracies:") print(f"Training Accuracy: {run_result['Train Accuracy']}") print(f"Test Accuracy: {run_result['Test Accuracy']}")
The Stochastic Gradient Descent Optimizer for the model has following accuracies: Training Accuracy: 53.125%
Test Accuracy: 62.5%
 
Data Pre-Processing and Data-Set splitting into train, test and validation
Description:
a. Settingdirectorynamesandcategories:
i. DataDirectory variable is set to the directory path where the images are stored.
  
 ii. Categories variable is a list of categories for the images, which in this case are "alpaca" and "not alpaca".
b. Loadinganddisplayinganimage:
i. A loop is initiated to iterate through each category in Categories.
ii. os.path.join() is used to create a path to the directory for each category.
iii. A loop is then initiated to iterate through each image in the category directory.
iv. cv2.imread() is used to read the image file into an array and convert it to grayscale.
v. plt.imshow() is used to display the grayscale image.
vi. plt.show() is used to display the image in a pop-up window.
vii. The loop is then broken after the first image is displayed.
c. Resizing an image:
i. The variable Img_Size is set to 28, which is the desired size for the images.
ii. cv2.resize() is used to resize the image to the desired size.
iii. plt.imshow() is used to display the resized image.
iv. plt.show() is used to display the image in a pop-up window.
d. Creatingtrainingdata:
i. A function called create_TrainData() is defined to load and preprocess all the images in the directories.
ii. A loop is initiated to iterate through each category in Categories.
iii. os.path.join() is used to create a path to the directory for each category.
iv. Categories.index() is used to assign a number to the class based on its index (0 for "alpaca" and 1 for "not alpaca").
v. Another loop is initiated to iterate through each image in the category directory.
vi. cv2.imread() is used to read the image file into an array and convert it to grayscale.
vii. cv2.resize() is used to resize the image to the desired size.
viii. The resized image array and its class number are appended to a list called Training_Data.
ix. If an exception is raised due to a distorted image, the loop is continued to the next image.
x. random.shuffle() is used to randomly shuffle the Training_Data list.
e. Splittingdataintofeaturesandlabels:
i. Two empty lists, X and y, are defined to hold the features and labels respectively.
ii. A loop is initiated to iterate through each feature-label pair in the Training_Data list.
iii. The feature array is appended to the X list, and the label is appended to the y list.
iv. np.array() is used to convert X and y to numpy arrays.

 CODE:
f. Preparing data for CNN model:
i. np.copy() is used to create a copy of the X numpy array.
ii. np.array().reshape() is used to reshape the X numpy array into a format suitable for the CNN model.
iii. tf.cast() is used to cast the X numpy array to a format supported by TensorFlow.
g. Splittingdataintotraining,validation,andtestingsets:
i. The size of the dataset is obtained using len().
ii. The total size of the validation and testing sets is calculated as one- third of the dataset size.
iii. The size of the validation set is calculated as half of the total validation and testing set size.
iv. The X_train and y_train arrays are created by slicing the X and y numpy arrays up to the size of the training set.
# Set the path to the directory containing the image dataset
DataDirectory = "./dataset"
# Set the categories of images to be classified Categories = ["alpaca",
"not alpaca"]
# Loop through each category of images
for category in Categories:
# Set the path to the directory containing the current category of
images
path = os.path.join(DataDirectory, category)
# Print the current directory path
print(path)
# Loop through each image in the current directory
for Image in os.listdir(path):
# Read the current image in grayscale mode using OpenCV img_arr = cv2.imread(os.path.join(path,Image),
cv2.IMREAD_GRAYSCALE)
# Display the current image using Matplotlib
plt.imshow(img_arr, cmap = "gray")
plt.show()
break
#Break out of the outer loop after showing images from the first
category only
break
# Resizing of the image
Image_Size = 28
New_Array = cv2.resize(img_arr, (Image_Size,Image_Size))
# Checking image distortion after resizing

 plt.imshow(New_Array, cmap = 'gray') plt.show()
# Initialising the training data
Training_Data = []
# deafining the load and preprocess of the images def create_TrainData():
for category in Categories:
# Path to alpaca or not alpaca directories
path = os.path.join(DataDirectory, category)
# Numbering classes based on their index, where alpaca is
represented by [0], rather than [1].
Class_Num = Categories.index(category) # Iterating through all the images
for Image in os.listdir(path): try:
# Reading and resizing the images to gray-scale
img_arr = cv2.imread(os.path.join(path,Image), cv2.IMREAD_GRAYSCALE)
New_Array = cv2.resize(img_arr, (Image_Size,Image_Size)) # Performing resizing operation
# Appending new array to the existing list
Training_Data.append([New_Array, Class_Num]) except Exception as e: pass create_TrainData()
import random random.shuffle(Training_Data)
X = [] # Features
y = [] # Labels
# Divinding the data into features and labels for features, labels in Training_Data:
X.append(features)
y.append(labels) X = np.array(X)
y = np.array(y)
# For CNN model
X_copy = np.copy(X)
# Re-shaping the images to get channel first format
X = np.array(X).reshape(-1, 1, Image_Size, Image_Size)
# Converting images into a format that is compatible with TensorFlow.
X = tf.cast(X, tf.float32)
#Establishing the sizes of sets to allocate 70% of the data for training and 30% for the remaining portion.
#Additionally, it is split into two subsets where 15% is allocated for validation and another 15% is set aside for testing.
dataset_len = len(X)
valid_test_size = int(dataset_len / 3) valid_size = int(valid_test_size / 2)
X_train = X[:(dataset_len - valid_test_size)] y_train = y[:(dataset_len - valid_test_size)]

X_valid = X[(dataset_len - valid_test_size):((dataset_len - valid_test_size) + valid_size)]
y_valid = y[(dataset_len - valid_test_size):((dataset_len - valid_test_size) + valid_size)]
X_test = X[((dataset_len - valid_test_size) + valid_size):] y_test = y[((dataset_len - valid_test_size) + valid_size):]
print(f"Total samples: {dataset_len}")
print(f"Training Dataset shape: X: {X_train.shape}, y: {y_train.shape}") print(f"Validation Dataset shape: X: {X_valid.shape}, y: {y_valid.shape}") print(f"Testing Dataset shape: X: {X_test.shape}, y: {y_test.shape}")
Total samples: 48
Training Dataset shape: X: (32, 1, 28, 28), y: (32,) Validation Dataset shape: X: (8, 1, 28, 28), y: (8,) Testing Dataset shape: X: (8, 1, 28, 28), y: (8,)

Outcomes using different Hyperparameters
Description:
a. To assess the performance of the model developed in task 3, various hyperparameters were trained on it.
b. Althoughthenumberofepochshadtoberestrictedto5and7,duetotheCustom Layer's computational complexity, the model was trained and assessed using a variety of optimizer configurations, learning rates, batch sizes, and learning rates to test for differences in convergence.
c. The Keras Optimizer class's Stochastic Gradient Descent and Adam optimizers are employed.



