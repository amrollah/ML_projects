clear all; clc

data = csvread('handout\training.csv');
X = data(:,1:27);
Y = data(:,28);

% cvpart = cvpartition(Y,'holdout',0.3);
% Xtrain = X(training(cvpart),:);
% Ytrain = Y(training(cvpart),:);
% Xtest = X(test(cvpart),:);
% Ytest = Y(test(cvpart),:);
penalty = [0, 5;1,0];

bag = fitensemble(X,Y,'Bag',400,'Tree',...
    'type','classification','cost', penalty);


%validation part
data = csvread('handout\validation.csv');

%figure;
%plot(loss(bag,Xtest,Ytest,'mode','cumulative'));
%xlabel('Number of trees');
%ylabel('Test classification error');

[predtest scores] = bag.predict(data);
f = fopen('predtest.txt', 'w');
for i=1:size(predtest, 1)
  fprintf(f, '%d\n', predtest(i));
end
fclose(f);

%score=norm(Ytest-predtest)/length(Ytest);
%scores_norm = norm_score(scores);
% scores_final = replaceonezero(scores(:,2));
% logloss(Ytest,scores_final)