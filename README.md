# VC Funding DNN Classification - UW FinTech Bootcamp Module 13 Challenge 

In this project I create and compare 4 deep neural networks trained to predict if a company will be successful if funded by a theoretical venture capital firm. These models are binary classification models that are evaluated on their accuracy to predict the yes/no values of if the company was successful after being funded. 

This is a TensorFlow project that I run on an M1 Mac, but it can be viewed in Google Colab! -> <a href="https://colab.research.google.com/github/ethansilvas/vc-funding-dnn-classification/blob/main/GC_venture_funding_with_deep_learning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

### Data Used
[applicants_data.csv](./Resources/applicants_data.csv) - Theoretical data of companies and their funding information; Label is the IS_SUCCESSFUL column

### Summary

I start by collecting the data then one-hot encoding the non-numeric data, split the data using scikitlearn's train_test_split(), and scaling the data using scikitlearn's StandardScaler()

![DataFrame showing one-hot encoded data with columns like status, ask_amt, and is_successful](./Resources/Images/encoded_dataframe.png)

I then define my first model that simply uses all features, two hidden layers with ReLU activation functions, a sigmoid output layer, and is evaluated using binary crossentropy with an adam optimizer. The number of nodes in each layer are defined by `(number_of_features + 1) // 2` and `(hidden_layer_1_nodes + 1) // 2`. Each of the models created are TensorFlow Sequential() models with Dense() layers. 

![TensorFlow output from using .summary() function showing two hidden layers with 58 and 29 outputs, and an output layer with 1 output](./Resources/Images/model1.png)

Then I attempt to optimize by first creating a model that is exactly the same as the first but instead has a third hidden layer with a node count of `(hidden_layer_2_nodes + 1) // 2`. 

![TensorFlow output from using .summary() function showing three hidden layers with 58, 29, and 15 outputs, and an output layer with 1 output](./Resources/Images/model2.png)

Next I create another model that is exactly like the first model but instead removes the SPECIAL_CONSIDERATIONS and STATUS features as they appear to be less predictive. 

![Outputs showing heavily imbalanced features with 30,000 counts of one value and 30 or less values for the other](./Resources/Images/bad_features.png)

Finally, I create a model that 

---

## Technologies

This is a Python 3.8 project ran in Google Colab but can be used in JupyterLab using a Conda dev environment. 

The following dependencies are used: 
1. [Jupyter](https://jupyter.org/) - Running code 
2. [Conda](https://github.com/conda/conda) (4.13.0) - Dev environment
3. [Pandas](https://github.com/pandas-dev/pandas) (1.3.5) - Data analysis
4. [Matplotlib](https://github.com/matplotlib/matplotlib) (3.5.1) - Data visualization
5. [Numpy](https://numpy.org/) (1.21.5) - Data calculations + Pandas support
6. [TensorFlow]

---

## Installation Guide

If you would like to run the program in JupyterLab, install the [Anaconda](https://www.anaconda.com/products/distribution) distribution and run `jupyter lab` in a conda dev environment.

To ensure that your notebook runs properly you can use the [requirements.txt](/Resources/requirements.txt) file to create an exact copy of the conda dev environment used in development of this project. 

Create a copy of the conda dev environment with `conda create --name myenv --file requirements.txt`

Then install the requirements with `conda install --name myenv --file requirements.txt`

---

## Usage

The Jupyter notebook []() will provide all steps of the data collection, preparation, and analysis. Data visualizations are shown inline and accompanying analysis responses are provided. It can be uploaded to Google Colab with the provided .csv files in `/Resources`, or can be viewed in the [GitHub uploaded version]()

---

## Contributors

[Ethan Silvas](https://github.com/ethansilvas)

---

## License

This project uses the [GNU General Public License](https://choosealicense.com/licenses/gpl-3.0/)