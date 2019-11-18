clc;
clear;
close all;
%5
file = load('data_3_5.csv');
x = file(:,1);
y=  file(:,2);
%a
[mu_x,sigma_x] = mle(x,'distribution','norm');
[mu_y,sigma_y] = mle(y,'distribution','norm');


%c
mdl = fitcnb(x,y);

scores = mdl.predict(x);
est = mdl.DistributionParameters;


[X,Y,T,AUC]  = perfcurve(y,scores,'2');
[X2,Y2,T2,AUC2]  = perfcurve(y,scores,'1');
figure();
plot(X2,Y2)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Naive Bayes Classifier Class 1')
figure();
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Naive Bayes Classifier Class 2')


%f
mu_a = -3;
sigma_a = 1;

norm_a = makedist('Normal',mu_a,sigma_a);
pdf_a = normpdf(x,norm_a.mean,norm_a.sigma);
pdf_a = pdf_a * 3/4.0;
mu_b = 7;
sigma_b =  .316;

norm_b = makedist('Normal',mu_b,sigma_b);
pdf_b = normpdf(x,norm_b.mean,norm_b.sigma);
pdf_b = pdf_b * 1/4.0;
cd = pdf_a + pdf_b;

[mu_cd,sigma_cd] = mle(cd,'distribution','Normal');