%% Clear stuff up
clc;
clear;
close all;

%% Setup data variables
file = table2array(    readtable("data.xlsx"));

exam_scores = file(:,1:3);

final_exam = file(:,4);

e1 = exam_scores(:,1);
e2 = exam_scores(:,2);
e3 = exam_scores(:,3);


%% A
fprintf("\n-------------PART A-------------\n");
%[B,dev,stats] = mnrfit(e1,final_exam);
x1 =  LinearModel.fit(e1,final_exam);
x2 = LinearModel.fit(e2,final_exam);
x3 = LinearModel.fit(e3,final_exam);

x1_pred = predict(x1,e1);
x2_pred = predict(x2,e2);
x3_pred = predict(x3,e3);

x1_acc = compareVecThresh(x1_pred,160)/25;
x2_acc = compareVecThresh(x2_pred,160)/25;
x3_acc = compareVecThresh(x3_pred,160)/25;
fprintf("x1 acc: %f\tX2 acc: %f\tX3 acc: %f\n",x1_acc,x2_acc,x3_acc);

%% B
fprintf("\n-------------PART B-------------\n");
x1x2 = LinearModel.fit([e1 e2],final_exam);
x1x3 = LinearModel.fit([e1 e3],final_exam);
x2x3 = LinearModel.fit([e2 e3],final_exam);

x1x2_pred = predict(x1x2,[e1 e2]);
x1x3_pred = predict(x1x3,[e1 e3]);
x2x3_pred = predict(x2x3,[e2 e3]);

x1x2_acc = compareVecThresh(x1x2_pred,160)/25;
x1x3_acc = compareVecThresh(x1x3_pred,160)/25;
x2x3_acc = compareVecThresh(x2x3_pred,160)/25;
fprintf("x1x2 acc: %f\tx1x3 acc: %f\tx2x3 acc: %f\n",x1x2_acc,x1x3_acc,x2x3_acc);

%% C
fprintf("\n-------------PART C-------------\n");

x1x2x3 = LinearModel.fit(exam_scores,final_exam);
bic =  log(x1x2x3.MSE) -1/2*x1x2x3.DFE*log(25);
x1x2x3_pred = predict(x1x2x3,exam_scores);
x1x2x3_acc = compareVecThresh(x1x2x3_pred,160)/25;
fprintf("x1x2x3 acc: %f\tBIC: %f\n",x1x2x3_acc,bic);

%% Helper Functions

[fList,plist] = matlab.codetools.requiredFilesAndProducts('HW4_P3.m');
plist.Name
function num_same = compareVec(vec1,vec2) 
    num_same = 0;
    for index = 1: length(vec1)
        if(vec1(index)==vec2(index))
            num_same = num_same+ 1;
        end
    end
end

function num_passed = compareVecThresh(vec1,thresh) 
    num_passed = 0;
    for index = 1: length(vec1)
        if(vec1(index)>=thresh)
            num_passed = num_passed+ 1;
        end
    end
end

function new_mat = norm_mat(mat)
    new_mat = zeros(length(mat),1);
    max_val =0;
    for i = 1:length(mat)
        if(mat(i)>=max_val)
            max_val=mat(i);
        end
    end
    for i = 1:length(mat)
        new_mat(i) = mat(i) / (max_val);
    end
    
end