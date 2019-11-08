clc;
%4a
file = load('assignment_2_problem_4.mat');
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
mle_4b_disease = mle(struct_4b_disease_data,'distribution','norm');
dist_4b_disease = makedist('Normal',mle_4b_disease(1),mle_4b_disease(2));
%Test Classifer against data
test_4b_disease = zeros(100,2);
count_correct = 0;
for index =1:100
    probability = pdf(dist_4b_disease,indicator(index));
    test_4b_disease(index) = probability;
    test_4b_disease(index,2) = patient(index);
    if(probability>.80 && patient(index)==1) 
        count_correct= count_correct+1;
    end
end
fprintf("Parameters for Discrete Unhealthy:\nMean: %f Variance: %f\n",mle_4b_disease(1),mle_4b_disease(2));
fprintf("Total correct estimations for discrete unhealthy: %d/60 (%f)\n\n",count_correct,count_correct/ 60);
%Build Healthy Classifer
struct_4b_healthy = struct_4a_healthy(any(struct_4a_healthy,2),:);
struct_4b_healthy_data = struct_4b_healthy(:,2);
mle_4b_healthy = mle(struct_4b_healthy_data,'distribution','norm');

dist_4b_healthy = makedist('Normal',mle_4b_healthy(1),mle_4b_healthy(2));
%Test Classifer against data
test_4b_healthy = zeros(100,2);
count_correct_healthy = 0;
for index =1:100
    probability = pdf(dist_4b_healthy,indicator(index));
    test_4b_healthy(index) = probability;
    test_4b_healthy(index,2) = patient(index);
    if(probability>.002 && patient(index)==0) 
        count_correct_healthy= count_correct_healthy+1;
    end
end
fprintf("Parameters for Discrete Healthy:\nMean: %f Variance: %f\n",mle_4b_healthy(1),mle_4b_healthy(2));
fprintf("Total correct estimations for discrete healthy: %d/40 (%f)\n\n",count_correct_healthy,count_correct_healthy/ 40);





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
classNames = [1,0];
prior = [.6 .4];
Mdl = fitcnb(naive_features,naive_y, 'ClassNames',classNames,'Prior',prior);
vals = Mdl.DistributionParameters;
naive_dist_a = makedist('Normal',vals{1}(1),vals{1}(2));
naive_dist_b = makedist('Normal',vals{1,2}(1),vals{1,2}(2));
naive_dist_c = makedist('Normal',vals{2}(1),vals{2}(2));
naive_dist_d = makedist('Normal',vals{2,2}(1),vals{2,2}(2));

test_naive_a = zeros(100,2);
test_naive_b = zeros(100,2);
test_naive_c = zeros(100,2);
test_naive_d = zeros(100,2);

for index =1:100
    probability_a = pdf(naive_dist_a,naive_x(index));
    probability_b = pdf(naive_dist_b,naive_x(index));
    probability_c = pdf(naive_dist_c,naive_x(index));
    probability_d = pdf(naive_dist_d,naive_x(index));
    %Best classifer
    test_naive_a(index) = probability_a;
    test_naive_a(index,2) = patient(index);
    test_naive_b(index) = probability_b;
    test_naive_b(index,2) = patient(index);

    test_naive_c(index) = probability_c;
    test_naive_c(index,2) = patient(index);

    test_naive_d(index) = probability_d;
    test_naive_d(index,2) = patient(index);


end
f= figure;
scatter(test_naive_a(1),test_naive_a(2));