clc; clear;

% Load training data
data = csvread('training.csv');

X = data(:,1:27);
Y = data(:,28);

kfolds = 2;
fSize=floor(size(X,1)/kfolds);

[C,gamma] = meshgrid((1:0.1:1.4), 0.45:0.01:0.55); %1:0.1:2

for j=(1:1:numel(C))
    err = zeros(kfolds,1);
    for i=1:kfolds
        xTest=X(((i-1)*fSize+1):(i*fSize), :);
        xTrain=X([1:((i-1)*fSize),(i*fSize+1):end], :);
        yTest=Y(((i-1)*fSize+1):(i*fSize));
        yTrain=Y([1:((i-1)*fSize),(i*fSize+1):end]);
        
       model = svmtrain(xTrain,yTrain,'kernel_function', 'rbf', 'rbf_sigma',gamma(j), 'method', 'SMO', 'boxconstraint', repmat(C(j),size(xTrain,1),1));

        prediction = svmclassify(model,X);
        err(i) = mean((Y-prediction).^2);
    end
    final_err(j) = mean(err);
 end

[min_err, idx] = min(final_err);

%best_gamma = 0.49;
best_gamma = gamma(idx);
best_c = C(idx);
model = svmtrain(X,Y,'kernel_function', 'rbf', 'rbf_sigma',best_gamma, 'method', 'SMO', 'boxconstraint', repmat(best_c,size(X,1),1));


%validation part
data = csvread('validation.csv');
predicta=zeros(4,size(data,1));

%% first method
predicta(1,:) = svmclassify(model, data);
%% Second method
bag = fitensemble(X,Y,'Bag',900,'Tree',...
    'type','classification');
[predicta(2,:) scores] = bag.predict(data);
% %% Third method
% %normalize_man2(X);
% predicta(3,:)=knnclassify(normalize_man2(data),normalize_man2(X),Y, 9);
% 
% %% Fourth method
% X=normalize_man2(X);
% d=normalize_man2(data);
% ens = fitensemble(X,Y,'AdaBoostM1',900,'Tree');
% predicta(4,:) = predict(ens,d);
%% prediction
prediction=mode(predicta(1:2,:));

f = fopen('val-pred.txt', 'w');
for i=1:size(prediction, 2)
  fprintf(f, '%d\n', prediction(1,i));
end
fclose(f);

%csvwrite('val-pred.txt', prediction);

 
%test section
data = csvread('testing.csv');
predicta=zeros(2,size(data,1));
predicta(1,:) = svmclassify(model, data);
[predicta(2,:) scores] = bag.predict(data);
prediction=mode(predicta(1:2,:));

f = fopen('test-pred.txt', 'w');
for i=1:size(prediction, 2)
  fprintf(f, '%d\n', prediction(1,i));
end
fclose(f);

%csvwrite('test-pred.txt', predict);