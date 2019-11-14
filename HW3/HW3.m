clc;
clear;
close all;
%5
file = load('data_3_5.csv');
x = file(:,1);
y=  file(:,2);

[mu_x,sigma_x] = mle(x,'distribution','norm');
[mu_y,sigma_y] = mle(y,'distribution','norm');

mdl = fitcnb(x,y);

est = mdl.DistributionParameters;

roc_data = roc(x,y);
roc_points = zeros(length(roc_data),2);
for index = 1:length(roc_data)
    ele  =cell2mat(roc_data(index));
    roc_points(index,1) =ele(1);
    roc_points(index,2) =ele(2);

end
plot(roc_points,'b');