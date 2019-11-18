%%
%Clear stuff up
clc;
clear;
close all;
%%
%Problem 5
file = load('data_3_5.csv');
x = file(:,1);
y=  file(:,2);
%Part A
[mu_x,sigma_x] = mle(x,'distribution','norm');
[mu_y,sigma_y] = mle(y,'distribution','norm');
fprintf("Feature x mle data: (%f, %f)\n", mu_x(1),mu_x(2));
fprintf("Feature y mle data: (%f, %f)\n", mu_y(1),mu_y(2));

%%
%Part B
%{
I believe the model estimators of both x and y fit the data given
%}
%%
%Part C
mdl = fitcnb(x,y);

scores = mdl.predict(x);
[X,Y,T,AUC]  = perfcurve(y,scores,'2');
[X2,Y2,T2,AUC2]  = perfcurve(y,scores,'1');
%figure(1);
subplot(2,1,1);

plot(X2,Y2)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Naive Bayes Classifier Class 1')
%figure(2);
subplot(2,1,2);

plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Naive Bayes Classifier Class 2')

%%
%Part D
%%
%Part f

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
fprintf("Likelihood of part f data: (%f, %f)\n",mu_cd(1),mu_cd(2));
%%
%Part g



%%
%Problem 6
file2 = load('data_3_6.csv');

x = file2(:,1);
y = file2(:,2);
z = file2(:,3);
%%
%Part A
figure(3);
scatter3(x,y,z);
xlabel("Feature 1");
ylabel("Feature 2");
zlabel("Class");
title("Feateures 1 & 2 with class");

figure(4);
scatter(x,z);
xlabel("Feature 1");
ylabel("Class");
title("Feateure 1 with class");

figure(5);
scatter(y,z);
xlabel("Feature 2");
ylabel("Class");
title("Feateure 2 with class");
%%
%Part B

class1_prior = sum(z(:) == 1) / length(z);
class2_prior = sum(z(:) == 2) / length(z);
class3_prior = sum(z(:) == 3) / length(z);

fprintf("Class 1 prior: %f\nClass 2 prior: %f\nClass 3 prior: %f\n",class1_prior,class2_prior,class3_prior);
%%
%Part C
p6_mdl = fitcnb([x,y],z);
isLabel_nb  = resubPredict(p6_mdl);
subplot(2,2,1);
naive_bayes_cm = confusionchart(z,isLabel_nb);
naive_bayes_err = loss(p6_mdl,[x,y],z);
title("Naive Bayes confusion matrix");

%%
%Part D
%%QDA
mdl_quad = fitcdiscr([x,y],z,'DiscrimType','quadratic');
isLabel_qda  = resubPredict(mdl_quad);
subplot(2,2,2);
qda_cm = confusionchart(z,isLabel_qda);
title("QDA confusion matrix");
qda_err = loss(mdl_quad,[x,y],z);

%%
%Part E
%LDA
mdl_linear = fitcdiscr([x,y],z);
isLabel_lda  = resubPredict(mdl_linear);
subplot(2,2,3);
lda_cm = confusionchart(z,isLabel_lda);
title("LDA confusion matrix");
lda_err = loss(mdl_linear,[x,y],z);

subplot(2,2,4);
bar([naive_bayes_err,qda_err,lda_err]);
ylabel("Error");
set(gca,'xticklabel',{"Naive Bayes","QDA","LDA"})
