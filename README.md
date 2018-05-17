# IntromlProject
Membership Inference Attacks Against Machine Learning Models
It is an attempt to reproduce and study the results published in the following paper as part of a class project for Introduction to Machine Learning class :
https://www.cs.cornell.edu/~shmat/shmat_oak17.pdf

CIFAR10_all_stages.ipynb contains the A to Z code for the experiments done on CIFAR 10 Dataset as part of this project.

--------------------------------------STAGE I-----------------------------------------  
Main aim: This part of code basically generates training and testing data for our attack model

Deatils: For each data_size in [2500, 5000, 10000, 15000] , we generates one target model and correspondingly 10 shadow models. Target model gives the test data for the attack model and shadow models give the training data for attack model. Data used in target models is totally disjoint from the data used for shadow models. The prediction vectors from these models are stored to be fed as input to the attack model. Note that target and shadow models are essentially clasification models built to classify CIFAR10 into 10 classes.

--------------------------------------STAGE II-----------------------------------------  
Main aim: This part of code basically creates, trains and tests our attack moddels for each data_size. 

Details: Authors of the paper wrote that the output statistics of the target model are different for each output class. Therefore they created 10 different classifier corresponding to each class of CIFAR10 as part of the attack model. So our attack model for a single data_size in whole comprises of 10 classifiers. We used SVM classifiers and selected the SVM parameters - C and gamma using five fold cross validation on training data of the attack models. Note that for a attack classifier of class i, only those prediction vectors are fed to it for which true label was i. This is how we split all the data among 10 attack classifers. All attack classifiers are basically binary classifiers trying to prediction whether a prediction vector came a sample who was part of the target model's training dataset or not.

--------------------------------------MITIGATION STAGE-----------------------------------------  
Main aim: To build a defense mechanism against the attack

Details: The attacks employed in this work exploit the fact that the networks are overfitted to the training data. Therefore as a defense mechanism we employ regularization so that overfitting in the target model reduces. This will make the target model less susceptible to these attacks
