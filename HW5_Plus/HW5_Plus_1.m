%% Clear stuff up
clc;
clear;
close all;

%% Load data

data = load('data_hw5_1.mat');
data = data.xyz;
input = data(:,1);
continous = data(:,2);
discrete = data(:,3);


%% Linear regression model

lin = LinearModel.fit(input,continous);

lin.