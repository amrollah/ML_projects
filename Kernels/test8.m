clc;clear

data = csvread('../handout/training.csv');
y=sqrt(data(:,15));
X = data(:,1:14);

incre1=2;  d1=2:incre1:8;
incre14=3; d14=9:incre14:36;
incre6=2; d6=2:incre6:16;
incre10=8; d10=8:incre10:32;

X=featureTransform(X,1,d1,incre1);
X=featureTransform(X,14,d14,incre14);
X=featureTransform(X,6,d6,incre6);
X=featureTransform(X,10,d10,incre10);

D = x2fx(X ,'interaction');
D(:,1) = []; 

DD=[D, log2(X(:,1:14)) sqrt(X(:,1:14)) ]; %log2(data(:,1:14)) sqrt(data(:,1:14)) data(:,1:14).^(1.3)


% newD = [];
% for i=1:ds
%     for j=i+1:ds
%         if P(i,j) < 0.01
%             corr_iy = corr(DD(:,i), y).^2;
%             corr_jy = corr(DD(:,j), y).^2;
%             if corr_iy > corr_jy
%                 newD = [newD, DD(:,j)];
%             else
%                 newD = [newD, DD(:,i)];
%             end
%         end        
%     end
% end
  
inx1=randperm(size(DD,2));
DD = DD(:, inx1);

% [Ry,Py]= corr(DD,y);
% [R,P]=corrcoef(DD);
% ds = size(R,2);
% a = sum(R)./Ry';
% maxVal=max(a);
% minVal=min(a);
% mean1=(minVal+maxVal)/2;
% 
% inx2=find(a > 0.05*mean1); %choose the good one
% DD=DD(:,inx2);

[~,~,~,model,~] = stepwisefit(DD,y,'inmodel',true(1,size(DD,2)));  %, 'penter', 0.05  
DD=DD(:,model);    

% h = 0:5e-5:20e-3;%5e-3;
h = 0:1e-5:3e-3;
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
    err(k)=sqrt((sum((yPred'-ytest).^2))/size(yPred,1))/mean(ytest);
    end

    errAvg=mean(err);
    
    if(errAvg<bestErr)
        bestErr=errAvg;
        bestW=W;
        bestH=h(i);
    end
    %disp(i)
    
end
fprintf('Error : %d\n', bestErr);
fprintf('H : %d\n', bestH);

data = csvread('../handout/validation.csv');

D1=featureTransform(data,1,d1,incre1);
D1=featureTransform(D1,14,d14,incre14);
D1=featureTransform(D1,6,d6,incre6);
D1=featureTransform(D1,10,d10,incre10);

D = x2fx(D1,'interaction');
D(:,1) = []; 

D=[D, log2(D1(:,1:14)) sqrt(D1(:,1:14)) ]; %log2(data(:,1:14)) sqrt(data(:,1:14)) data(:,1:14).^(1.3)
D = D(:, inx1);
% D = D(:,inx2);
D = D(:,model);    

Dp=[ones(size(D,1),1)  D];
yPred=abs((bestW'*Dp')').^2;
% yPred=2.^(bestW'*Dp')';

f = fopen('ypred.txt', 'w');
for i=1:size(yPred, 1)
  fprintf(f, '%f\n', yPred(i));
end
fclose(f);
