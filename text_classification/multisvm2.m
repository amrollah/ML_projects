function [result] = multisvm2(trainData,trainLabel,testData)
%# train one-against-all models
path(path, 'libsvm-3.172\matlab');

numTest = size(testData,1);
numLabels = size(trainLabel,1);

model = cell(numLabels,1);
testLabel = zeros();
for k=1:numLabels
    model{k} = svmtrain(double(trainLabel==k), trainData, '-c 1 -g 0.2 -b 1');
end

%# get probability estimates of test instances using each model
prob = zeros(numTest,numLabels);
for k=1:numLabels
    [~,~,p] = svmpredict(double(testLabel==k), testData, model{k}, '-b 1');
    prob(:,k) = p(:,model{k}.Label==1);    %# probability of class==k
end

%# predict the class with the highest probability
[~,pred] = max(prob,[],2);
acc = sum(pred == testLabel) ./ numel(testLabel);    %# accuracy
C = confusionmat(testLabel, pred);                %# confusion matrix

result = pred;
end