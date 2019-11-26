%%
%Clear stuff up
clc;
clear;
close all;
%% A
X = [3, 5, 7, 18, 43, 85, 91, 98, 100, 130, 230, 487];
log_sample_means = zeros(1000,1);
sample_means = zeros(1000,1);
for iter = 1:1000
    s = datasample(X,length(X));
    log_sample_means(iter) = log(mean(s));
    sample_means(iter) = mean(s);
    
end
%% B
estimated_vals = mle(log_sample_means,'distribution','Lognormal');
fprintf("Estimated mean for log of sample means: %f\n",estimated_vals(1));
%% C
stderror = std(log_sample_means) / sqrt(length(log_sample_means));
fprintf("Standard error: %f\n",stderror);
%% D

logmean_sample_mean = log(mean(sample_means));
log_data_mean = log(mean(X));
fprintf("Difference of menas: %f - %f -> %f\n",logmean_sample_mean,log_data_mean, logmean_sample_mean-log_data_mean);
%% E
N = 1000;                                      % Number of ‘Experiments’ In Data Set
yMean = logmean_sample_mean;                                    % Mean Of All Experiments At Each Value Of ‘x’
ySEM = stderror;                              % Compute ‘Standard Error Of The Mean’ Of All Experiments At Each Value Of ‘x’
CI95 = tinv([0.025 0.975], N-1);                    % Calculate 95% Probability Intervals Of t-Distribution
yCI95 = bsxfun(@times, ySEM, CI95(:));              % Calculate 95% Confidence Intervals Of All Experiments At Each Value Of ‘x’

fprintf("95%% confidence interval: (%f,%f)\n",yCI95(1),yCI95(2));

[fList,plist] = matlab.codetools.requiredFilesAndProducts('HW4_P3.m');
plist.Name