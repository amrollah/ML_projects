clc; clear; close all

%% load training data
data = csvread('../handout/training.csv');

% form possible features
% myd = data;
% myd(:, 2:5) = data(:,2:5)/8;
% myd(:, 6) = data(:, 6)/2;
% myd(:, 14) = data(:, 14)/3;
% myd(:,[8:9,11:13]) = log2(data(:,[8:9,11:13]));
% d=[data(:,1:14), log(data(:,1:14)), sqrt(data(:,1:14)),...
%     data(:,3).^3,...      % Feature 3 by power of 3 (potentially also pw2)
%       data(:,5).^3,...      % Feature 5 by power of 3
%       data(:,12).^3,...     % Feature 12 by power of 3 
%       data(:,13).^4,...     % Feature 13 by power of 4 (or pw3 or pw5)
% ];

% d=[data(:,1:14),data(:,2:5)/8, data(:, 6)/2, log(data(:,7:14)), ... %data(:,1:14).^2,
%     data(:,1:14).^3, data(:,1:14).^4, log(data(:,1:14)/log(5)),...
%     log(data(:,1:14))/log(3),log(data(:,1:14))/log(4),log2(data(:,1:14))];
d = [data(:,1)/2,...         %
      data(:,2:5)/8,...     % Divide by increment
      data(:, 6)/2,...      % Divide by increment
      data(:,[7,10]),...      %
      log2(data(:,[8:9,11:13])),...
      data(:,14)/3,...      %   
      data(:,[1,2:5,6,8,9,11,13,14]),...      
      data(:,1:14).^2,...
      log(data(:,1:14)),...
      sqrt(data(:,1:14))
      ];

dn = d;
% siz = size(data,2)-1;
siz = size(d,2);
for i=1:siz-1
    for j=i+1:siz   
            dn=[dn, d(:,i).* d(:,j)];
    end
end
d = dn;
 
% ind=randperm(size(d,1));
% d=d(ind,:);


[r c]=size(d);
y=data(:,15);

% corres = corrcoef([d,y]);
% n = size(d,2);
% imagesc(abs(corres(end,:))); % plot the matrix
% set(gca, 'XTick', 1:n); % center x-axis ticks on bins
% set(gca, 'YTick', 1:n); % center y-axis ticks on bins
% colormap('jet'); % set the colorscheme


siz = size(d,2);
% for i=500:530
%     figure(i);
%     plot(d(:,i),y,'.b');
% end

%normalization
m=mean(d);
d=d-repmat(m,r,1);
sigma=std(d);
d=(d)./repmat(sigma,r,1);

% m=mean(y);
% y=y-repmat(m,r,1);
% sigma=std(y);
% y=(y)./repmat(sigma,r,1);

% hyper=0.2:0.5:3; % for closed form
hyper = 1e-7:5e-5:10e-4; % for ridge
%initial values
fold=5;
fSize=floor(r/fold);
maxErr=Inf;
bestModel=[];
bestW=[];
kfoldErr=zeros(1,fold);
opts = statset('display','iter');
i=1;
% for i=1:size(hyper,2)
%     [b,se,pval,inmodel,stats,nextstep,history] = stepwisefit(d,y,'display', 'on');
%     fun = @(xtrain,ytrain,xtest,ytest)(rmse(ridge(ytrain, xtrain,hyper,0)'*xtest')', ytest);
%     fun = @(xtrain,ytrain,xtest, ytest)LinRegress(xtrain,ytrain,xtest, ytest, hyper(i));
%     inmodel = sequentialfs(fun,d,y, 'direction', 'forward', 'cv', 5, 'options', opts);
    [betahat1,se1,pval1,inmodel,stats1] = stepwisefit(d,y,'inmodel',true(1,size(d,2)));
    dd=d(:,inmodel);
for i=1:size(hyper,2)     
%     for k=1:fold
%         xTest=dd(((k-1)*fSize+1):(k*fSize), :);
%         xTrain=dd([1:((k-1)*fSize),(k*fSize+1):end], :);
%         yTest=y(((k-1)*fSize+1):(k*fSize));
%         yTrain=y([1:((k-1)*fSize),(k*fSize+1):end]);
%         
%         %test the accurcay of this hyperparameter-not exactly true, better suggestion?
%         err(k)= LinRegress(xTrain, yTrain, xTest, yTest, hyper(i));  %we add one again in the function
%     end
    err= LinRegress(dd, y, dd, y, hyper(i));  %we add one again in the function
    err = err/size(y,1);
    %compute W for this model
    dd = [dd, ones(size(dd,1),1)];
    W=(dd'*dd + hyper(i) * eye(size(dd, 2)) )\(dd'*y);
        
    errAvg(i)=err;
    if(errAvg(i) < maxErr)
        bestHyper=hyper(i);
         bestModel=inmodel;
         bestW=W;
        maxErr=errAvg(i);
    end
    
end

save W bestW;

fprintf('Error : %d\n', maxErr);
%% load validation set
data = csvread('../handout/validation.csv');

%choose the same features
% d=[data(:,1:14),  data(:,2:5)/8, data(:, 6)/2, log(data(:,7:14)), ...  %data(:,1:14).^2,
%     data(:,1:14).^3, data(:,1:14).^4, log(data(:,1:14)/log(5)),...
%     log(data(:,1:14))/log(3),log(data(:,1:14))/log(4),log2(data(:,1:14))];

d = [data(:,1)/2,...         %
      data(:,2:5)/8,...     % Divide by increment
      data(:, 6)/2,...      % Divide by increment
      data(:,[7,10]),...      %
      log2(data(:,[8:9,11:13])),...
      data(:,14)/3,...      %   
      data(:,[1,2:5,6,8,9,11,13,14]),...      
      data(:,1:14).^2,...
      log(data(:,1:14)),...
      sqrt(data(:,1:14))
      ];
%   data(:,3).^3,...      % Feature 3 by power of 3 (potentially also pw2)
%       data(:,5).^3,...      % Feature 5 by power of 3
%       data(:,12).^3,...     % Feature 12 by power of 3 
%       data(:,13).^4,...     % Feature 13 by power of 4 (or pw3 or pw5)
  
% data(:, 2:5) = data(:,2:5)/8;
% data(:, 6) = data(:, 6)/2;
% data(:, 14) = data(:, 14)/3;
% data(:,[8:9,11:13]) = log2(data(:,[8:9,11:13]));
% d=[data(:,1:14), log(data(:,1:14)), sqrt(data(:,1:14)),...
%     data(:,3).^3,...      % Feature 3 by power of 3 (potentially also pw2)
%       data(:,5).^3,...      % Feature 5 by power of 3
%       data(:,12).^3,...     % Feature 12 by power of 3 
%       data(:,13).^4,...     % Feature 13 by power of 4 (or pw3 or pw5)
% ];


dn = d;
siz = size(d,2);
for i=1:siz-1
    for j=i+1:siz   
            dn=[dn, d(:,i).* d(:,j)];
    end
end
d = dn;

%normalization
m=mean(d);
d=d-repmat(m,r,1);
sigma=std(d);
d=(d)./repmat(sigma,r,1);

data=d(:,bestModel);

%prediction
x = [data, ones(size(data, 1), 1)];
yPred=abs((bestW'*x')');

f = fopen('ypred.txt', 'w');
for i=1:size(yPred, 1)
  fprintf(f, '%f\n', yPred(i));
end
fclose(f);

%% Notes
%Note for me : How to pass more than four arguments to a sequentialfs
%n = 1;
%criterion_function = @(xtrain, ytrain, xtest, ytest)
%myRealCriterionFunction(xtrain, ytrain, xtest, ytest, n);
%[selected_fs, history] = sequentialfs(criterion_function, ...
