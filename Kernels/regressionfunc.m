% Performs ridge reqression and establishes best hyperparameter through
% cross validation. n is number of times regression should be applied
% and data is data to perform the regression on,
% returns mean error and error standard deviation
function [meanError, errorStd] = regressionfunc(n, input)

error = zeros(n,1);

for run=1:n
    
    data = input;
    
    ind=randperm(size(data,1));
    data=data(ind,:);

    [r, c]=size(data);

    % %normalization
    %average
    m=mean(data);
    %standard deviation
    sigma=std(data);
    %normalized data
    dataN=(data-repmat(m,r,1))./repmat(sigma,r,1);

    %features
    x=dataN(:, 1:(c-1));
    %lables
    y=data(:,c);


    %Number of folds
    Kfold=5;
    %Size of each fold
    fSize=floor(r/Kfold);

    %hyperParams
    Lambda= 0 : 0.1: 30;
    hyperParams = Lambda;
    iErr= zeros(1,Kfold);
    hErr=zeros(1,size(hyperParams,1));

    for h=1:size(hyperParams, 2)

        for i=1:Kfold
            xTest=x(((i-1)*fSize+1):(i*fSize), :);
            xTrain=x([1:((i-1)*fSize),(i*fSize+1):end], :);
            yTest=y(((i-1)*fSize+1):(i*fSize));
            yTrain=y([1:((i-1)*fSize),(i*fSize+1):end]);

            iErr(1, i)=LinRegress(xTrain, yTrain, xTest, yTest, hyperParams(1, h));
        end

        hErr(h) = mean(iErr);
    end


    [val, ~] = min(hErr);
    
    error(run) = val;
end

meanError = mean(error);
errorStd = std(error);

end