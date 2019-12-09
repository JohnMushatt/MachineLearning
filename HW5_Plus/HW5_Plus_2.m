%% Clear stuff up
clc;
clear;
close all;

%% Load Data
data = load('data_hw5_2.mat');
data = data.xy;

inputs = data(:,1:5);
output = data(:,6);

%% Linear model
lin_lin = LinearModel.fit(inputs,output);
lin = fitrlinear(inputs,output);

MSE = loss(lin,inputs,output);
fprintf("RSS value: %f\n",lin_lin.RMSE);
fprintf("R^2 value: %f\n",lin_lin.Rsquared.Ordinary);

