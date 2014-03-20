clear;
addpath('libsvm-3.17\matlab');

data = csvread('handout\training.csv'); % read a csv file
features = data(:, 1:27);
labels = data(:, 28); % labels from the 1st column

m = size(features,1);
Max = repmat(max(features),m,1);
Min = repmat(min(features),m,1);
features = (features - Min)./(Max - Min);

% adding some new features
newfeatures = [log(features + eps), sqrt(features)];
% Normalization
m = size(newfeatures,1);
Max = repmat(max(newfeatures),m,1);
Min = repmat(min(newfeatures),m,1);
newfeatures = (newfeatures - Min)./(Max - Min);

features = [features, newfeatures];

delim = 147;
Xtrain = features(1:delim,:);
Xtest = features(delim:end,:);

Ltrain = labels(1:delim,end);
Ltest = labels(delim:end,end);

Xtrain_sparse = sparse(Xtrain); % features must be in a sparse matrix
libsvmwrite('dataLibsvm.train', Ltrain, Xtrain_sparse);

Xtest_sparse = sparse(Xtest); % features must be in a sparse matrix
libsvmwrite('dataLibsvm.test', Ltest, Xtest_sparse);


% then run this comomand in shell (from too classification directory)
% python fselect.py ..\..\dataLibsvm.train ..\..\dataLibsvm.test
