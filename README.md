# applied-machine-learning
This is a repository of multiple applied machine learning & Data Science projects.

**Contents**
- Projects
    1. Data EDA & Visualization
    2. Supervised Learning
    3. Neural Networks

**About**</br>
This Repository had 3 projects for Data EDA, applied machine learning & deep learning.

## Projects
### 1. Data EDA & Visualization
This is an Exploratory Data Analysis and Visualization (EDAV) project.
This project involves Exploring data, summarizing their main characteristics, often with visual to form better questions.
The 2 datasets used for this project are from [The Museum of Modern Art (MoMA) Collection](https://github.com/MuseumofModernArt/collection).
    1. The Artworks dataset
    2. The Artists dataset

The goal of this project is to allow myself to work with data, explore it, and visualize it to tell a story.
as part of my exploratory data analysis and visualization (EDAV), I was able to:
* check the missing values in all columns. paying attention that not all missing values will be `NaN`s. Making sure that non-nan values are valid.
* check data type of all columns and make sure they correspond with what I expected them to be. dates to be of `datetime` format, numbers to be of either `int` or `float`.
* plot distribution of numeric columns and histogram of categorical columns.
* calculate and visualize top 5 nationalities of artists
* calculate and visualize top 5 nationalities of artists by gender
* merge two dataframes using `ConstituentID` as the key
* pick a nationality and filter the merged data to find all artworks done by artists of that nationality. Figured out how to handle artworks that have multiple artists.
* Do additional analysis of the data that picked my interest.

---

### 2. Supervised Learning
In this project, I build a **red wine quality classifier** using supervised machine learning techniques.
The dataset used in this project is the [Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality) from the UCI Machine Learning Repository.

I conducted the following processess to built the classifier:

1. **Exploratory data analysis:** checked columns to understand datatypes, value ranges, find possible missing values, the correlation between features, and create relevant data visualization
2.  **Modeling**:
    * created a sklearn pipeline  that included preprocessing steps such as scaling and one hot encoding of categorical variable.
    *  trained **Linear learners** (logistic regression & SVM) classfiers and reported the performance using confusion matrix
    *  trained **non-linear classifers** (random forrest & SVM with non linear kernels) and reported the performance using confusion matrix.
3. **Hyperparameter Tuning**: Picked the model with the highest F1 score and ran a ```grid search``` to improve the result. Reported the best set of parameters.

4. Used the best model from step 3 to create a function that takes a single row of test data and predict the quality of wine. The function is described below:

```python
# Function to predict the wine quality
def wine_quality(row:np.array, model:Pipeline) -> int:
    """
    Input:
        row: np array of test feature data
        model: trained model pipeline
    Output:
        result: wine quality as int
    """

    result = model.predict(row)[0]

    return result
```

---

### 3. Neural Networks
This project has two sections. Each section has been broken down into multiple parts.

* First part of this project is building a `wine quality cliassifier` again with the same `wine quality` data, using a neural net classifier with 2 dense hidden layers and reporting the classifier performance for the test set using confusion matrix.
    - Preprocessing the data (scaling, encoding, and handling missing values)
    - Using 16 and 8 parameters parameters for hidden layers. Using Adam optmizer without any dropout. Use ReLU for the hidden layers and softmax for the output layer activation function.
    - Doubled the number of parameters while keeping everything else the same and compared the result. Explaining the difference in the performance of both the classifiers.
    ```python
    # Funtion to build a custom NN classifier
    def two_layer_NN(normalizer, layer_1_neurons, layer_2_neurons, hidden_activation, 
                        output_neurons, output_activation, optimizer, loss, metrics):
        model = tf.keras.Sequential([
        normalizer,
        layers.Dense(layer_1_neurons, activation=hidden_activation),
        layers.Dense(layer_2_neurons, activation=hidden_activation),
        layers.Dense(output_neurons, activation=output_activation)
        ])

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        return model
    ```

* The second part of this porject is to classify images using CNN. The dataset for this excersise is [food images](https://www.kaggle.com/kmader/food41). 
    - Built a custom model with my choice of the parameters like layer size, number of convolution layer, filter size, etc.
    ```python
    # Function to built the custom CNN image classifier model
    def build_custom_CNN(num_filters, filter_size, conv_activation, input_shape, 
                            max_pool_size, layer_1_neurons, layer_2_neurons, output_neurons):
        model = tf.keras.Sequential([
            layers.Conv2D(num_filters, filter_size, activation=conv_activation, input_shape=input_shape),
            BatchNormalization(),
            layers.MaxPooling2D(max_pool_size),
            layers.Conv2D(num_filters, filter_size, activation=conv_activation),
            BatchNormalization(),
            layers.MaxPooling2D(max_pool_size),
            layers.Conv2D(num_filters, filter_size, activation=conv_activation),
            BatchNormalization(),
            layers.MaxPooling2D(max_pool_size),
            layers.Conv2D(num_filters, filter_size, activation=conv_activation),
            BatchNormalization(),
            layers.MaxPooling2D(max_pool_size),
            layers.Conv2D(num_filters, filter_size, activation=conv_activation),
            BatchNormalization(),
            layers.MaxPooling2D(max_pool_size),
            layers.Flatten(),
            layers.Dense(layer_1_neurons, activation=hidden_activation),
            layers.Dense(layer_2_neurons, activation=hidden_activation),
            layers.Dense(output_neurons, activation=output_activation)
        ])

        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=True)

        return model
    ```
    - Then, took advantage of `transfer learning` by using `EfficientNetB0` as my model. Compared the result and explained the difference with the custom model.
    ```python
    # Functionto create the transfer learning CNN classifier model
    def transfer_CNN_model(drop_out_rate, max_pool_size, num_classes, 
                                optimizer, loss, metrics, input_shape):
        inputs = layers.Input(shape=input_shape)
        x = inputs
        
        model = EfficientNetB0(include_top=False, 
                            input_tensor=x, 
                            weights="imagenet")
        
        # Freezing the pretrained weights
        model.trainable = False

        # Rebuilding top layers
        x = layers.MaxPooling2D(max_pool_size)(model.output)
        x = layers.BatchNormalization()(x)

        x = layers.Dropout(drop_out_rate, name="top_dropout")(x)
        x = layers.Flatten()(x)
        outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

        # Compile
        model = tf.keras.Model(inputs, outputs, name="EfficientNet")
        optimizer = optimizer
        model.compile(
            optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=True
        )
        return model
    ```
