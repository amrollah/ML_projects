clc; clear; close all

%% load training data
data = csvread('../handout/training.csv');

y = data(:,15);
X = data(:,1:14);
yd = [log2(y), sqrt(y)];

incre1=2;  d1=2:incre1:8;
incre14=3; d14=9:incre14:36;
incre6=2; d6=2:incre6:16;
incre10=8; d10=8:incre10:32;

X=featureTransform(X,1,d1,incre1);
X=featureTransform(X,14,d14,incre14);
X=featureTransform(X,6,d6,incre6);
X=featureTransform(X,10,d10,incre10);

bd2 = [data(:,1:7),log2(data(:,8:9)),data(:,10), log2(data(:,11:13)), data(:, 14)];
bd3 = [X(:,1:7),log2(X(:,8:9)),X(:,10), log2(X(:,11:13)), X(:, 14)];
      
bd = data(:,1:14);
d = [bd, X, bd3, log(bd)]; 
  

ds = size(d,2);
ys = size(yd,2);

corres = zeros(ds,ys);
for i=1:ds
    for j=1:ys
        corres(i, j) = corr(d(:,i), yd(:,j)).^2;
    end
end


imagesc(corres); % plot the matrix
set(gca, 'XTick', 1:ds); % center x-axis ticks on bins
set(gca, 'YTick', 1:ys); % center y-axis ticks on bins
set(gca, 'XTickLabel', (1:ds)); % set y-axis labels;
title('Correlations', 'FontSize', 14); % set title
colormap('jet'); % set the colorscheme
% colorbar on; % enable colorbar

