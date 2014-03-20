%score 0.1803
clc;clear

data = csvread('training.csv');
y=sqrt(data(:,15));
X = data(:,1:14);

D = x2fx(X ,'quadratic');
D(:,1) = []; % No constant term

DD=[D, log2(data(:,1:14)) sqrt(data(:,1:14))];

corres = zeros(size(DD,2),1);
for i=1:size(DD,2)
        [rho(i) pval(i)] = corr(DD(:,i), y);
end

inx=find(pval < 0.05);
DD=DD(:,inx);
  
[~,~,~,model,~] = stepwisefit(DD,y,'inmodel',true(1,size(DD,2)));    
DD=DD(:,model);

DD=[DD log(data(:,13)) log(data(:,13)).^2 sqrt(log(data(:,13))) data(:,13)];

h = 0:1e-5:20e-3;%5e-3;
bestErr=Inf;
fold=10;
[r c]=size(DD);
fSize=floor(r/fold);
err=[];

for i=1:length(h)

    for k=1:fold
    xtest=DD(((k-1)*fSize+1):(k*fSize), :);
    xtrain=DD([1:((k-1)*fSize),(k*fSize+1):end], :);
    ytest=y(((k-1)*fSize+1):(k*fSize));
    ytrain=y([1:((k-1)*fSize),(k*fSize+1):end]);
        
    W = ridge(ytrain,xtrain,h(i),0);
    xtest=[ones(size(xtest,1),1)  xtest];
    yPred=(W'*xtest');
    
    Mean=sum(ytest)/size(ytest,1);
    err(k)=sqrt((sum((yPred'-ytest).^2))/size(yPred,1))/Mean;
    end

    errAvg=sum(err)/length(err);
    
    if(errAvg<bestErr)
        bestErr=errAvg;
        bestW=W;
        bestH=h(i);
    end
%    disp(i)
    
end

data = csvread('testing.csv');
D = x2fx(data,'quadratic');
D(:,1) = []; 

D=[D, log2(data(:,1:14)) sqrt(data(:,1:14)) ];
D=D(:,inx);  
D=D(:,model);    

D=[D log(data(:,13)) log(data(:,13)).^2 sqrt(log(data(:,13))) data(:,13)];
Dp=[ones(size(D,1),1)  D];
yPred=abs((bestW'*Dp')').^2;


f = fopen('ypred.txt', 'w');
for i=1:size(yPred, 1)
  fprintf(f, '%f\n', yPred(i));
end
fclose(f);
