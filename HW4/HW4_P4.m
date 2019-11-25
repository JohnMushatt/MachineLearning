%% Clear stuff up
clc;
clear;
close all;

%% Setup data variables
file = load("data_3_6.csv");
c1 = file(file(:,3)==1);
c2 = file(file(:,3)==2);
c3 = file(file(:,3)==3);

features = file(:,1:2);
labels = file(:,3);
%% Naive Bayes with 10-fold validation
fold_size = length(features)/ 10;
training_size = fold_size *9;
testing_size = fold_size;

randomized_data = file(randperm(size(file, 1)), :);
training_data_features = randomized_data(1:training_size,1:2);
training_data_labels= randomized_data(1:training_size,3);

testing_data_features = randomized_data(training_size+1:end,1:2); 
testing_data_labels = randomized_data(training_size+1:end,3); 

nb = fitcnb(training_data_features,training_data_labels);
acc_tr_nb = 1-loss(nb,training_data_features,training_data_labels);

fprintf("Naive Bayes Training accuracy: %f\n",acc_tr_nb);

acc_te_nb = 1-loss(nb,testing_data_features,testing_data_labels);
fprintf("Naive Bayes Testing accuracy: %f\n",acc_te_nb);

%% QDA with 10-fold validation
randomized_data = file(randperm(size(file, 1)), :);
training_data_features = randomized_data(1:training_size,1:2);
training_data_labels= randomized_data(1:training_size,3);

testing_data_features = randomized_data(training_size+1:end,1:2); 
testing_data_labels = randomized_data(training_size+1:end,3); 

qda = fitcdiscr(training_data_features,training_data_labels,'DiscrimType','quadratic');
acc__tr_qda = 1-loss(qda,training_data_features,training_data_labels);

fprintf("QDA Training accuracy: %f\n",acc__tr_qda);

acc_te_qda = 1-loss(qda,testing_data_features,testing_data_labels);
fprintf("QDA Testing accuracy: %f\n",acc_te_qda);
 

%% LDA with 10-fold validation

randomized_data = file(randperm(size(file, 1)), :);
training_data_features = randomized_data(1:training_size,1:2);
training_data_labels= randomized_data(1:training_size,3);

testing_data_features = randomized_data(training_size+1:end,1:2); 
testing_data_labels = randomized_data(training_size+1:end,3); 

lda = fitcdiscr(training_data_features,training_data_labels,'DiscrimType','linear');

acc_tr_lda = 1 - loss(lda,training_data_features,training_data_labels);
fprintf("LDA Training accuracy: %f\n",acc_tr_lda);

acc_te_lda = 1 - loss(lda,testing_data_features,testing_data_labels);
fprintf("LDA Testing accuracy: %f\n", acc_te_lda);
