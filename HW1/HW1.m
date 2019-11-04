
%---Problem 7---
ratings = load('jester_ratings.dat');
data = ratings(:,3);
graph = histogram(data,'Normalization','pdf');
hold on;
g2 = histogram(data,100);
g2.Normalization = 'probability';
%%extract parameters
counts = graph.Values;
sum_counts = sum(counts);
width = graph.BinWidth;
%%area of the histogram
area = sum_counts*width;


%---Problem 8---
mle_normal = mle(data,'distribution','norm');
mean_norm = mle_normal(1);
variance_normal = mle_normal(2);
mean_and_variance = sprintf("MLE Mean: %f\nMLE Variance: %f",mean_norm,variance_normal);
text(0,.1,mean_and_variance);


%---Problem 9---

%Scaled beta distribution
%beta_params = betafit(data);
data_beta = fitdist(data,'Beta');
%mle_beta = mle(data,'distribution','Beta');
%mean_beta = mle_beta(1);
%variance_beta = mle_beta(2);
%Logistic distribution
%data_logistic = fitdist(data,'Logistic');
mle_logistic = mle(data,'distribution','logistic');
mean_logistic = mle_logistic(1);
variance_logistic = mle_logistic(2);

%---Problem 10---


%test = kfolds(data,10);
function [test, train] = kfolds(data, k)

  n = size(data,1);

  test{k,1} = [];
  train{k,1} = [];

  chunk = floor(n/k);

  test{1} = data(1:chunk,:);
  train{1} = data(chunk+1:end,:);

  for f = 2:k
      test{f} = data((f-1)*chunk+1:(f)*chunk,:);
      train{f} = [data(1:(f-1)*chunk,:); data(f*chunk+1:end, :)];
  end
end