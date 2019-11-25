%% Clear stuff up
clc;
clear;
close all;

%% Setup data variables
file = table2array(    readtable("data.xlsx"));

exam_scores = file(:,1:3);
final_exam = file(:,4);

%% 5 Linear Regression

%% A
fprintf("\n-------------PART A-------------\n");

x1 = LinearModel.fit(exam_scores(:,1),final_exam);
x2 = LinearModel.fit(exam_scores(:,2),final_exam);
x3 = LinearModel.fit(exam_scores(:,3),final_exam);
RSSx1 = x1.Rsquared.Ordinary;
RSSx2 = x2.Rsquared.Ordinary;
RSSx3 = x3.Rsquared.Ordinary;

x1_pred = predict(x1,exam_scores(:,1))
x2_pred = predict(x2,exam_scores(:,2))
x3_pred = predict(x3,exam_scores(:,3))

%fprintf("X1 acc: %f\tX2 acc: %f\tX3 acc: %f\n",x1_score,x2_score,x3_score);
fprintf("RSS for x1: %f\nRSS for x2: %f\nRSS for x3: %f\n",RSSx1,RSSx2,RSSx3 );

%% B
fprintf("\n-------------PART B-------------\n");

x1x2_y = LinearModel.fit(exam_scores(:,1:2),final_exam);
RSSx1x2 = x1x2_y.Rsquared.Ordinary;
x1x2_pred = predict(x1x2_y,exam_scores(:,1:2))
x1x3_y = LinearModel.fit([exam_scores(:,1) exam_scores(:,3)],final_exam);
RSSx1x3 = x1x3_y.Rsquared.Ordinary;
x1x3_pred = predict(x1x3_y,[exam_scores(:,1) exam_scores(:,3)])

x2x3_y = LinearModel.fit(exam_scores(:,2:3),final_exam);
RSSx2x3 = x2x3_y.Rsquared.Ordinary;
x2x3_pred =predict(x2x3_y,exam_scores(:,2:3))

fprintf("\n-----------------------\nRSS for x1x2: %f\nRSS for x1x3: %f\nRSS for x2x3: %f\n",RSSx1x2,RSSx1x3,RSSx2x3 );


%% C
fprintf("\n-------------PART C-------------\n");
x1x2x3 = LinearModel.fit(exam_scores,final_exam);
RSSx1x2x3 = x1x2x3.Rsquared.Ordinary;
x1x2x3_pred = predict(x1x2x3,exam_scores)
compareVecMean(x1x2x3_pred,final_exam);
fprintf("\n-----------------------\nRSS for x1x2x3: %f\n",RSSx1x2x3);


Resx1x2x3 = x1x2x3.Residuals;
figure(1);
x1x2x3.plotResiduals;

%% D
fprintf("\n-------------PART D-------------\n");

fprintf("X1\n");
compareVecMean(x1_pred,final_exam);
fprintf("X2\n");
compareVecMean(x2_pred,final_exam);
fprintf("X3\n");
compareVecMean(x3_pred,final_exam);

fprintf("X1 and X2\n");
compareVecMean(x1x2_pred,final_exam);
fprintf("X1 and X3\n");
compareVecMean(x1x3_pred,final_exam);
fprintf("X2 and X3\n");
compareVecMean(x2x3_pred,final_exam);

fprintf("X1, X2, and X3\n");
compareVecMean(x1x2x3_pred,final_exam);

%% Helper functions

function num_same = compareVec(vec1,vec2) 
    num_same = 0;
    for index = 1: length(vec1)
        if(vec1(index)==vec2(index))
            num_same = num_same+ 1;
        end
    end
end

function mean_difference = compareVecMean(vec1,vec2)
    mean_difference = zeros(length(vec1),1);
    for index = 1:length(vec1)
        mean_difference(index) = vec1(index) - vec2(index);
        
    end
    mean_difference = mean(mean_difference);
    v1_mean = mean(vec1);
    v2_mean = mean(vec2);
    v1_var = var(vec1);
    v2_var = var(vec2);
    fprintf("|\tMean Difference\t  |\tV1 Mean\t  |\tV2 Mean\t  |\tV1 Variance\t  |\tV2 Variance\t  |\n");
    fprintf("\t\t%e\t\t%f\t%f\t%f\t\t%f\n",mean_difference,v1_mean,v2_mean,v1_var,v2_var);
end