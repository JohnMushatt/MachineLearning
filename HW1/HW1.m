
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

normalized_data = normalize(data,'range');

%---Problem 8---
mle_normal = mle(normalized_data,'distribution','norm');
mean_norm = mle_normal(1);
variance_normal = mle_normal(2);
mean_and_variance = sprintf("MLE Mean: %f\nMLE Variance: %f",mean_norm,variance_normal);
text(0,.1,mean_and_variance);


%---Problem 9---
%Scaled beta distribution
beta_dist = fitdist(normalized_data,'Beta');
mle_beta = mle(normalized_data,'distribution','beta');
mean_beta = mle_beta(1);
variance_beta = mle_beta(2);

alpha = 1.2404;
beta = 0.9265;
beta_xvals = -10:.01:10;
scaled_beta=Scaled_BetaPDF(beta_xvals, alpha, beta, -10, 10);
plot(beta_xvals, scaled_beta, 'LineWidth', 2);
%Logistic distribution
temp = normalized_data;
indices= find(temp <= 0);
temp(indices) = [];
ab = any(temp <=0);

data_logistic = fitdist(temp,'Lognormal');
mle_logistic = mle(temp,'distribution','Lognormal');
mean_logistic = mle_logistic(1);
variance_logistic = mle_logistic(2);

%---Problem 10---


[m,n] = size(normalized_data) ;

P = 0.90 ;



%normal
log_data_normal = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
count = 1;
for test = 1:10
    idx = randperm(m)  ;
    Training = normalized_data(idx(1:round(P*m)),:) ;
    Testing = normalized_data(idx(round(P*m)+1:end),:) ;
    current_mle = mle(Training,'distribution','norm');
    
    mean_norm = current_mle(1);
    
    variance_normal = current_mle(2);
    
    current_pdf = makedist("Normal",mean_norm,variance_normal);
    log_val = 0.0;
    for index = 1:length(Testing)
        
        current_prob = pdf(current_pdf,Testing(index));
        log_prob = log(current_prob);
        
        log_val= log_val + log_prob;
    end
    log_val = log_val / length(Testing);
    fprintf("Current log value for normal dist: %d at index: %d\n",log_val,count);

    log_data_normal{count} = log_val;
    count = count + 1;
end
%beta
log_data_beta = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
count = 1;
for test = 1:10
    idx = randperm(m)  ;
    Training = normalized_data(idx(1:round(P*m)),:) ;
    Testing = normalized_data(idx(round(P*m)+1:end),:) ;
    current_mle = mle(Training,'distribution','Beta');
    
    mean_beta = current_mle(1);
    
    variance_beta = current_mle(2);
    
    current_pdf = makedist("Beta",mean_beta,variance_beta);
    log_val = 0.0;
    for index = 1:length(Testing)
        
        current_prob = pdf(current_pdf,Testing(index));
        log_prob = log(current_prob);
        
        log_val= log_val + log_prob;
    end
    log_val = log_val / length(Testing);
    fprintf("Current log value for normal beta: %d at index: %d\n",log_val,count);

    log_data_beta{count} = log_val;
    count = count + 1;
end
%log
log_data_log = {0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0};
count = 1;
[m,n] = size(temp) ;

for test = 1:10
    idx = randperm(m)  ;
    Training = temp(idx(1:round(P*m)),:);
    Testing = temp(idx(round(P*m)+1:end),:) ;
    current_mle = mle(Training,'distribution','Lognormal');
    
    mean_log = current_mle(1);
    
    variance_log = current_mle(2);
    
    current_pdf = makedist("Lognormal",mean_log,variance_log);
    log_val = 0.0;
    for index = 1:length(Testing)
        
        current_prob = pdf(current_pdf,Testing(index));
        log_prob = log(current_prob);
        
        log_val= log_val + log_prob;
    end
    log_val = log_val / length(Testing);
    fprintf("Current log value for normal log: %d at index: %d\n",log_val,count);

    log_data_log{count} = log_val;
    count = count + 1;
end
figure;
hold on;
cellfun(@plot,log_data_normal,log_data_beta,log_data_log);

function PDF = Scaled_BetaPDF(y, a, b, p, q)
PDF = ( (y-p).^(a-1) .* (q - y).^(b-1) ) ./ ( (q - p).^(a+b-1) .* beta(a,b) );
end
