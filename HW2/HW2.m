clc;
clear;
close all;

p3_data = transpose([2.0 -3.6 2.2 4.9 1.5 3.1 2.2 -.9 .9 -2.4; 
    7.89 -16.55 6.73 17.91 2.06 12.84 8.13 -5.35 3.97 -12.31]);
mle_a_top=0;
mle_a_bot  =0;
mle_b=0;
v_data = var(p3_data);
variance_val=0;
for index =1:length(p3_data)
    val_x = p3_data(index);
    val_y = p3_data(index,2);
    mle_a_top = mle_a_top + val_x;
    mle_a_bot= mle_a_bot+ val_x^2;
    mle_b = mle_b +val_y -val_x;
    
    variance_val = -length(p3_data) / val_x + (val_y - (val_x)) / (val_x)^3;
end

mle_a = mle_a_top / mle_a_bot;
fprintf("MLE_A = %f\tMLE_B = %f\t Variance = %f\n",mle_a,mle_b,variance_val);
figure();
lin_data = computeLinModel(mle_a,mle_b,p3_data);
lin_plot = lin_data(:,2);
plot(p3_data(:,1),p3_data(:,2),'+',p3_data(:,1),lin_plot);
%3




%4a
file = load('assignment_2_problem_4.mat');
temp_data = file.xy;
indicator = file.xy(:,1);
cont_measure = file.xy(:,2);
patient = file.xy(:,3);

struct_4a_disease= zeros(100,3);

for index = 1:100
    if(patient(index)==1)
        struct_4a_disease(index,1) = indicator(index);
        struct_4a_disease(index,2) = cont_measure(index);
        struct_4a_disease(index,3) = patient(index);
    end
end
hist_disease = histogram(struct_4a_disease);
hold on;
struct_4a_healthy= zeros(100,3);
for index = 1:100
    if(patient(index)==0)
        struct_4a_healthy(index,1) = indicator(index);
        struct_4a_healthy(index,2) = cont_measure(index);
        struct_4a_healthy(index,3) = patient(index);
    end
end
hist_healthy = histogram(struct_4a_healthy);
%4b
%Build Disease Classifer
struct_4b_disease = struct_4a_disease(any(struct_4a_disease,2),:);
struct_4b_disease_data = struct_4b_disease(:,1);
mle_4b_disease = mle(struct_4b_disease_data,'distribution','bernoulli');
dist_4b_disease = makedist('Binomial',100,mle_4b_disease);
%Test Classifer against data
test_4b_disease = zeros(100,2);
count_correct = 0;
for index =1:100
    probability = pdf(dist_4b_disease,indicator(index));
    test_4b_disease(index) = probability;
    test_4b_disease(index,2) = patient(index);
    if(probability>1.372033444959369e-65 && patient(index)==1) 
        count_correct= count_correct+1;
    end
end
fprintf("Parameters for Binomial Unhealthy:\nProbability P: %f\n",mle_4b_disease);
fprintf("Total correct estimations for Binomial unhealthy: %d/60 (%f)\n\n",count_correct,count_correct/ 60);
%Build Healthy Classifer
struct_4b_healthy = struct_4a_healthy(any(struct_4a_healthy,2),:);
struct_4b_healthy_data = struct_4b_healthy(:,1);
mle_4b_healthy = mle(struct_4b_healthy_data,'distribution','bernoulli');

dist_4b_healthy = makedist('Binomial',100,mle_4b_healthy);
%Test Classifer against data
test_4b_healthy = zeros(100,2);
count_correct_healthy = 0;
for index =1:100
    probability = pdf(dist_4b_healthy,indicator(index));
    test_4b_healthy(index) = probability;
    test_4b_healthy(index,2) = patient(index);
    if(probability <2.323555148909575e-20 && patient(index)==0) 
        count_correct_healthy= count_correct_healthy+1;
    end
end
fprintf("Parameters for Binomial Healthy\nProbability P: %f\n",mle_4b_healthy);
fprintf("Total correct estimations for Binomial healthy: %d/40 (%f)\n\n",count_correct_healthy,count_correct_healthy/ 40);





%4c
%Build Disease Classifer
struct_4c_disease = struct_4a_disease(any(struct_4a_disease,2),:);
struct_4c_disease_data = struct_4b_disease(:,2);
mle_4c_disease = mle(struct_4c_disease_data,'distribution','norm');
dist_4c_disease = makedist('Normal',mle_4c_disease(1),mle_4c_disease(2));
%Test Classifer against data
test_4c_disease = zeros(100,2);
count_correct = 0;
for index =1:100
    probability = pdf(dist_4c_disease,cont_measure(index));
    test_4c_disease(index) = probability;
    test_4c_disease(index,2) = patient(index);
    if(probability>.35 && patient(index)==1) 
        count_correct= count_correct+1;
    end
end
fprintf("Parameters for Continous Unhealthy:\nMean: %f Variance: %f",mle_4c_disease(1),mle_4c_disease(2));
fprintf("\nTotal correct estimations for continous unhealthy: %d/60 (%f)\n\n",count_correct,count_correct/ 60);

%Build Healthy Classifer
struct_4c_healthy = struct_4a_healthy(any(struct_4a_healthy,2),:);
struct_4c_healthy_data = struct_4c_healthy(:,2);
mle_4c_healthy = mle(struct_4c_healthy_data,'distribution','norm');
dist_4c_healthy = makedist('Normal',mle_4c_healthy(1),mle_4c_healthy(2));
%Test Classifer against data
test_4c_healthy = zeros(100,2);
count_correct = 0;
for index =1:100
    probability = pdf(dist_4c_healthy,cont_measure(index));
    test_4c_healthy(index) = probability;
    test_4c_healthy(index,2) = patient(index);
    if(probability >.6 && patient(index)==0) 
        count_correct= count_correct+1;
    end
end
fprintf("Parameters for Continous Healthy:\nMean: %f Variance: %f\n",mle_4c_healthy(1),mle_4c_healthy(2));

fprintf("Total correct estimations for continous healthy: %d/40 (%f)\n",count_correct,count_correct/ 40);


%4d

naive_features = file.xy(:,1:2);
naive_y = file.xy(:,3);
naive_cov = cov(naive_features);
classNames = [1,0];
prior = [.6 .4];
Mdl = fitcnb(naive_features,naive_y, 'ClassNames',classNames,'Prior',prior);
vals = Mdl.DistributionParameters;
res = Mdl.predict(naive_features);
naive_count =0;
for index = 1:100
    if(res(index)==patient(index))
        naive_count = naive_count +1;
    end
end
fprintf("Naive results: %d / 100 (%f)\n",naive_count,naive_count/100);


%5
prior = [1/3 1/3 1/3];
mu_1 = [0,0];
mu_2 = [1,1];
mu_3 = [-1,1];

sigma_1 = [0.7 0;0 .7];    
sigma_2 = [ 0.8 .2;.2 .8 ];
sigma_3 = [ 0.8 .2;.2 .8 ];
sigma_total = [sigma_1,sigma_2,sigma_3];
x1 = -3:0.2:3;
x2 = -3:0.2:3;
[X1,X2] = meshgrid(x1,x2);
X = [X1(:) X2(:)];
test1 = [-.5 .5];
test2 = [.5 .5];

sigma_1_x1 = mvnpdf(test1,mu_1,sigma_1);
sigma_1_x2 = mvnpdf(test2,mu_1,sigma_1);

sigma_2_x1 = mvnpdf(test1,mu_2,sigma_2);
sigma_2_x2 = mvnpdf(test2,mu_2,sigma_2);

sigma_3_x1 = mvnpdf(test1,mu_3,sigma_3);
sigma_3_x2 = mvnpdf(test2,mu_3,sigma_3);

result_x1 = max([sigma_1_x1,sigma_2_x1,sigma_3_x1]);
result_x2 = max([sigma_1_x2,sigma_2_x2,sigma_3_x2]);
fprintf("x1 is class 1 with P(%f)\nx2 is class 2 with P(%f)\n",result_x1,result_x2);



%e1=-1/2 * log(abs(sigma_1)) - 1/2*(transpose(x1-mu_1))*(sigma_1^-1)*(x1-mu_1) + log(prior(1));
%6
%-------------------------
%see heightweight.m
%-------------------------

%7
class1_mu = [1;-1];
class1_sigma = [2 0;0 16];
class1_samples = mvnrnd(class1_mu,class1_sigma,400);

class2_mu = [-1;1];
class2_sigma = [1 0; 0 1];
class2_samples = mvnrnd(class2_mu,class2_sigma,300);

class3_mu = [3;3];
class3_sigma = [4 1; 1 2];
class3_samples = mvnrnd(class3_mu,class3_sigma,300);
p6 = figure();
hold on;
plot(1:length(class1_samples),class1_samples,'g');
plot(1:length(class2_samples),class2_samples,'r');
plot(1:length(class3_samples),class3_samples,'b');
cat1_data = [class1_samples;class2_samples;class3_samples];
cat1_mean = mean(cat1_data);
cat1_cov = cov(cat1_data);

p7_mvn = mvnrnd(cat1_mean,cat1_cov,1000);   
p7 = figure();
hold on;
plot(1:length(class1_samples),class1_samples,'g');
plot(1:length(class2_samples),class2_samples,'r');
plot(1:length(class3_samples),class3_samples,'b');
plot(1:length(p7_mvn),p7_mvn,'c');

function points = computeLinModel(a,b,d) 
    points = zeros(length(d),2);
    for index = 1:length(d)
        points(index) = d(index);
        points(index,2) = a * points(index) + b;
    end
end
