clc;
clear;
close all;
%% Linear/ quadratic discriminant analysis for Height Weight data
%
%%

% This file is from pmtk3.googlecode.com

clear all
rawdata = loadData('heightWeight');
data.Y = rawdata(:,1); % 1=male, 2=female
data.X = [rawdata(:,2) rawdata(:,3)]; % height, weight


MdlLinear = fitcdiscr(data.X,data.Y);

mean_data = mean(data.X);
menaclass = predict(MdlLinear,mean_data);

maleNdx = find(data.Y == 1);
femaleNdx = find(data.Y == 2);
classNdx = {maleNdx, femaleNdx};

% Plot class conditional densities
for tied=[false true]
    figure;
    colors = 'br';
    sym = 'xo';
    styles = {'bx', 'ro'};
    for c=1:2
        X = data.X(classNdx{c},:);
        % fit Gaussian
        mu{c}= mean(X);
        if tied
            Sigma{c} = cov(data.X); % all classes
        else
            Sigma{c} = cov(X); % class-specific
        end
        str = sprintf('%s%s', sym(c), colors(c));
        % Plot data and model
        h=scatter(X(:,1), X(:,2), 100, str); %set(h, 'markersize', 10);
        hold on;
        [x,y] = meshgrid(linspace(50,80,100), linspace(80,280,100));
        [m,n]=size(x);
        X = [reshape(x, n*m, 1) reshape(y, n*m, 1)];
        g{c} = reshape(gaussProb(X, mu{c}(:)', Sigma{c}), [m n]);
        contour(x,y,g{c}, colors(c));
    end
    xlabel('height'); ylabel('weight')
    % Draw decision boundary
    for c=1:2
        [cc,hh]=contour(x,y,g{1}-g{2},[0 0], '-k');
        set(hh,'linewidth',3);
    end
    if tied
        title('tied covariance')
        printPmtkFigure(sprintf('heightWeightLDA'))
    else
        title('class-specific covariance')
        printPmtkFigure(sprintf('heightWeightQDA'))
    end
end

[cc,hh]=contour(x,y,g{1} - g{2},[0 0], '-k');
contour_points = hh.ContourMatrix();
%contour_points = (2:end);
contour_x = hh.ContourMatrix(1,2:end);
contour_y = hh.ContourMatrix(2,2:end);
qda = spline(contour_y,contour_x);
test_data = rawdata(:,2);
predicted = ppval(qda,test_data);
test_results = zeros(210,1);
ccount=0;
for index =1:length(predicted) 
    element = predicted(index);
    actual_result = rawdata(index,3);
    if element >= actual_result || index>200
        test_results(index)= 1;
    else 
        test_results(index) = 2;
    end
    %res = abs(act_weight-element);
   	%ccount = ccount+ res < 2;
    op1 = test_results(index);
    op2 =rawdata(index,1);
    if(test_results(index) ~= rawdata(index,1))
        ccount= ccount+ 1;
    end
        
end
figure();
scatter(1:210,test_results);
hold on;
error_rate = ccount / 210;
fprintf("Error rate for qda/lda: %f",error_rate);

