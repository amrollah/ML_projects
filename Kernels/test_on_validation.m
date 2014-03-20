%% load validation set
data = csvread('../handout/validation.csv');
load('params.mat', 'bestW', 'bestModel', 'bestHyper');

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

%   d = x2fx(d ,'quadratic');
  
dn = d;
siz = size(data,2);
for i=1:siz-1
    for j=i+1:siz   
%         dn=[dn, d(:,i).* d(:,j)];
            dn=[dn, data(:,i).* data(:,j)];
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
disp('finished');