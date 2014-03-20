function  err=LinRegress(xTrain, yTrain, xTest, yTest, hyper)
   
    %Method 1: closed form solution 
%     xTrain=[ xTrain, ones(size(xTrain, 1), 1)];
%     W=(xTrain'*xTrain + hyper * eye(size(xTrain, 2)) )\(xTrain'*yTrain);
        
    %Method 2: least squre linear regression (doesn't work well)
%    W = lsqlin(xTrain, yTrain, [], []);
    
    %Method 3: ridge
     W = ridge(yTrain,xTrain,hyper,0);
     xTest=[ones(size(xTest, 1), 1), xTest];
    
%     xTest=[ xTest, ones(size(xTest, 1), 1)];
    yPred=(W'*xTest');
    
%     err = sum((yPred'-yTest).^2);%/size(yTest,1); % Mean squared error
    err=sqrt((sum(yPred'-yTest).^2)/numel(yPred));%/mean(yTest); % CVRMSE

end
