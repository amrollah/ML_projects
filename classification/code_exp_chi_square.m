%score 0.1803
clc;clear;

addpath('libsvm-3.17/matlab');
% addpath('libsvm_chi_ksirg\matlab');

data = csvread('handout/training.csv');

% Permute for cross-validation and split into features and labels
ind=randperm(size(data,1));
data=data(ind,:);
X = data(:,1:27);
Y = data(:,28);

% normalize data to [0,1] range
%mn = min(X,[],1); mx = max(X,[],1);
%X = bsxfun(@rdivide, bsxfun(@minus, X, mn), mx-mn);

%# grid of parameters
folds = 5;
fSize=floor(size(X,1)/folds);
[C,gamma] = meshgrid(2.^(1:2:10), -2:1:2);

%# grid search, and cross-validation
cv_acc = zeros(numel(C),1);
for j=1:numel(C)
    err = ones(folds, 1);
    for i=1:folds
        xTest=X(((i-1)*fSize+1):(i*fSize), :);
        xTrain=X([1:((i-1)*fSize),(i*fSize+1):end], :);
        yTest=Y(((i-1)*fSize+1):(i*fSize));
        yTrain=Y([1:((i-1)*fSize),(i*fSize+1):end]);
        
        K = [(1:size(xTrain,1))', expChi2Kernel(xTrain, xTrain, gamma(j))];
        KK = [(1:size(xTest,1))', expChi2Kernel(xTest, xTest, gamma(j))];

        model = svmtrain(yTrain, K, ...
                    sprintf('-c %f -w1 1 -w-1 5 -t 4 -h 0 -m 1024', C(j)));
                
        [Lpredict, acc, ~] = svmpredict(yTest, KK, model);
               
        err(i) = mean((Lpredict - yTest).^2);
    end
    
    cv_acc(j) = mean(err);
end

%# pair (C,gamma) with best accuracy = smallest error
[~,idx] = min(cv_acc);

%# contour plot of paramter selection
contour(C, gamma, reshape(cv_acc,size(C))), colorbar
hold on
plot(C(idx), gamma(idx), 'rx')
text(C(idx), gamma(idx), sprintf('Acc = %.2f %%',cv_acc(idx)), ...
    'HorizontalAlign','left', 'VerticalAlign','top')
hold off
xlabel('log_2(C)'), ylabel('log_2(\gamma)'), title('Cross-Validation Accuracy')

%# now you can train you model using best_C and best_gamma
best_C = C(idx);
best_gamma = gamma(idx);

K = [(1:size(X,1))', expChi2Lernel(X, X, best_gamma)];
model = svmtrain(Y, K, sprintf('-c %f -w1 1 -w-1 5 -t 4 -h 0 -m 1024', best_C));


data = csvread('handout/validation.csv');

% normalize data to [0,1] range
%mn = min(data,[],1); mx = max(data,[],1);
%data = bsxfun(@rdivide, bsxfun(@minus, data, mn), mx-mn);

KK = [(1:size(data,1))', expChi2Kernel(data, data, best_gamma)];

%fake labels just to get svmpredict working!
Vlabel = ones(size(data,1),1);

[Lpredict, ~, ~] = svmpredict(Vlabel, KK, model);

f = fopen('Lpred.txt', 'w');
for i=1:size(Lpredict, 1)
  fprintf(f, '%d\n', Lpredict(i));
end
fclose(f);
