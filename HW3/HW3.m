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
data = [x y];
%Part A
[mu_x,sigma_x] = mle(x,'distribution','norm');
dat = mle(x,'distribution','norm');
[mu_y,sigma_y] = mle(y,'distribution','norm');
fprintf("Feature x mle data: (%f, %f)\n", mu_x(1),mu_x(2));
fprintf("Feature y mle data: (%f, %f)\n", mu_y(1),mu_y(2));

%%
%Part B
%{
I believe the model estimators do not give a good fit because of the
distribution being multimodal
%}
idx = find(y ==1);
idy = find(y ==2);
x_class1 = zeros(length(idx),1);
x_class2 = zeros(length(idy),1);
for index = 1:length(idx)
    x_class1(index) = x(idx(index));
end
for index = 1:length(idy)
    x_class2(index) = x(idy(index));
end

col = find(idx);
figure(1);
subplot(1,2,1);
histogram(x_class1);
subplot(1,2,2);
histogram(x_class2);
%%
%Part C
mdl = fitcnb(x,y);

scores = mdl.predict(x);
[X,Y,T,AUC]  = perfcurve(y,scores,'2');
[X2,Y2,T2,AUC2]  = perfcurve(y,scores,'1');
figure(2);
subplot(2,1,1);

plot(X2,Y2)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Naive Bayes Classifier Class 1')
subplot(2,1,2);

plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Naive Bayes Classifier Class 2')

%%
%Part D
figure(3);
t = find(y==1);
xpoints_1 = x(t );
ypoints = y;
dat = margin(mdl,x,y);
dat = [x y ];
%% build ROC emp
ROC_curve=[0 0];
for i=-10:0.01:10
    conf_mat_ROC=zeros(2);
    for j=1:length(dat)
        if(dat(j,1)<i)
            if(dat(j,2)==1)
                conf_mat_ROC(1,1)=conf_mat_ROC(1,1)+1;
            else
                conf_mat_ROC(1,2)=conf_mat_ROC(1,2)+1;
            end
        else
            if (dat(j,2)==2)
                conf_mat_ROC(2,2)=conf_mat_ROC(2,2)+1;
            else
                conf_mat_ROC(2,1)=conf_mat_ROC(2,1)+1;
            end
        end
    end
    TPR=conf_mat_ROC(1,1)/(conf_mat_ROC(1,1)+conf_mat_ROC(2,1));
    FPR=conf_mat_ROC(1,2)/(conf_mat_ROC(2,2)+conf_mat_ROC(1,2));
    ROC_curve=[ROC_curve;[FPR TPR]];
end
hold on
plot(ROC_curve(:,1),ROC_curve(:,2));
title("ROC Curve empircal");
hold off
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
figure(4);
histogram(cd);
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
figure(5);
scatter3(x,y,z);
xlabel("Feature 1");
ylabel("Feature 2");
zlabel("Class");
title("Feateures 1 & 2 with class");

figure(6);
scatter(x,z);
xlabel("Feature 1");
ylabel("Class");
title("Feateure 1 with class");

figure(7);
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


indices = crossvalind('kfold',x,5);
val = [x y];
updated_val = [x y z];
cp_naive = classperf(z);
acc = zeros(4,1);
for i = 1:5
    %{
    updated_val = updated_val(randperm(size(updated_val, 1)), :);
    x = updated_val(:,1);
    y = updated_val(:,2);
    z = updated_val(:,3);
    train_x = zeros(length(find(indices~=5)),1);
    train_y = zeros(length(find(indices~=5)),1);
    train_z = zeros(length(find(indices~=5)),1);
    test_x = zeros(length(find(indices==5)),1);
    test_y = zeros(length(find(indices==5)),1);
    test_z = zeros(length(find(indices==5)),1);
    

    for index = 1:length(find(indices~=5))
        if indices(index) ~=5
            train_x(index) = x(index);
            train_y(index) = y(index);
            train_z(index) = z(index);
        elseif indices(index)==5

        end
    end
    for index = 1:length(find(indices==5))
        if indices(index) ==5
            test_x(index) = x(index);
            test_y(index) = y(index);
            test_z(index) = z(index);
        end
    end
    zero_index = find(test_x==0);
    test_x(zero_index) = [];
    
    zero_index = find(train_x==0);
    
    train_x(zero_index) = [];
    
    zero_index = find(test_y==0);
    test_y(zero_index) = [];
    
    zero_index = find(train_y==0);
    train_y(zero_index) = [];
    
    %}
    P = 0.80 ;

    [m,n] =size(val);
    idx = randperm(m)  ;
    Training = updated_val(idx(1:round(P*m)),:) ;
    Testing = updated_val(idx(round(P*m)+1:end),:) ;
    p6_mdl = fitcnb([Training(:,1) Training(:,2)],Training(:,3));
    
    res = p6_mdl.predict([Testing(:,1),Testing(:,2)]);
    tp=0;tn=0;fp=0;fn=0;
    for index = 1:length(res)
        test_a = res(index);
        test_r = Testing(index);
        fprintf("%d,%d\n",test_a,test_r);
        if res(index) ==1 & Testing(index)==1
            tp = tp +1;
        elseif res(index) ==1 & Testing(index)==2
            fp = fp+1;
        elseif res(index) == 2 & Testing(index)==2
            tn = tn+1;
        elseif res(index) ==2 & Testing(index)==1
            fn = fn+1;
        end
    end
    acc = (tp + tn) / (tp + fn + fp + tn);
    fprintf("Current accuracy: %f\n",acc);

end
p6_mdl = fitcnb([x y],z);

isLabel_nb  = resubPredict(p6_mdl); 


figure(8);
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
