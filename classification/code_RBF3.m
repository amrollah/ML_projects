%0.24
clc;clear
addpath('C:\Users\Rabeeh\Desktop\Machine learning\Exercises\project\classification\code\libsvm-3.17\windows');

data = csvread('training.csv');
ind=randperm(size(data,1));
data=data(ind,:);

X = data(:,1:27);
X=X+repmat(max(max(abs(X))),size(X));
X=[X, sqrt(X)];

Y = data(:,28);
% 
% %%
% f = @(X,Y)oobError(TreeBagger(50,X,Y,'method','classification','oobpred','on'),'mode','ensemble');
% opt = statset('display','iter');
% [fs,history] = sequentialfs(f,X,Y,'options',opt,'cv','none');
% %%
load fs;
X=X(:,fs);

X=normalize_man2(X);



%# grid of parameters
folds =5;
[C,gamma] = meshgrid(-15:1:15, -15:1:15);
%[C, gamma] = meshgrid(.02:.05:.4, 15:2:30);

%# grid search, and cross-validation
cv_acc = zeros(numel(C),1);
kerneltype=2;
for i=1:numel(C)
    cv_acc(i) = svmtrain(Y, X, ...
                    sprintf('-c %f -w1 1 -w-1 5 -t 7 -g %f -v %d  -t %d', 2^C(i), 2^gamma(i), folds, kerneltype));
%     cv_acc(i) = svmtrain(Y, X, ...
%                     sprintf('-c %f -w1 1 -w-1 5 -t 7 -g %f -v %d  -t %d', C(i), gamma(i), folds, kerneltype));
    
    sprintf('Iteration: %d', i);
end

%# pair (C,gamma) with best accuracy
[~,idx] = max(cv_acc);

%# contour plot of paramter selection
figure();
contour(C, gamma, reshape(cv_acc,size(C))), colorbar
hold on
plot(C(idx), gamma(idx), 'rx')
text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
    'HorizontalAlign','left', 'VerticalAlign','top')
hold off
xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')

%# now you can train you model using best_C and best_gamma
best_C = 2^C(idx);
best_gamma = 2^gamma(idx);

model = svmtrain(Y, X, sprintf('-c %f -w1 1 -w-1 5 -g %f', best_C, best_gamma));
boxplot(cv_acc);

data = csvread('validation.csv');
X = data(:,1:27);
X=X+repmat(max(max(abs(X))),size(X));
X=[X, sqrt(X)];
X=X(:,fs);



% X = x2fx(X ,'quadratic');
% X(:,1) = []; % No constant term
% m = size(X,1);
% Max = repmat(max(X),m,1);
% Min = repmat(min(X),m,1);
% X = (X - Min)./(Max - Min);
X=normalize_man2(X);

%X=normalize(X);
%fake labels just to get svmpredict working!
Vlabel = ones(size(data,1),1);

[Lpredict, ~, ~] = svmpredict(Vlabel, X, model);

f = fopen('Lpred.txt', 'w');
for i=1:size(Lpredict, 1)
  fprintf(f, '%d\n', Lpredict(i));
end
fclose(f);




% test on testing set
data = csvread('testing.csv');
X=data;
X=X+repmat(max(max(abs(X))),size(X));
X=[X, sqrt(X)];
X=X(:,fs);

%fake labels just to get svmpredict working!
Vlabel = ones(size(X,1),1);
data=normalize_man2(X);
% % Normalization
% m = size(data,1);
% Max = repmat(max(data),m,1);
% Min = repmat(min(data),m,1);
% data = (data - Min)./(Max - Min);
% Lpredict_test = svmpredict_approx(data, approxmodel, ProbClassify);
[Lpredict, ~, ~] = svmpredict(Vlabel, data, model);

f = fopen('Lpred_test.txt', 'w');
for i=1:size(Lpredict, 1)
  fprintf(f, '%d\n', Lpredict(i));
end
fclose(f);
