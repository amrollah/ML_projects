clc; clear; close all

data = csvread('../handout/validation.csv');
load W

% Create features
data=[data(:,1),...             %
      data(:,2:5)/8,...         % Divide by increment
      data(:, 6)/2,...          % Divide by increment
      data(:,7:14),...          %
      data(:,3).^3,...          % Feature 3 by power of 3 (potentially also pw2)
      data(:,5).^3,...          % Feature 5 by power of 3
      data(:,12).^3,...         % Feature 12 by power of 3 
      data(:,13).^4,...         % Feature 13 by power of 4 (or pw3 or pw5)
      log(data(:,3))/log(5),... % Feature 3 log of base 5
      log(data(:,4))/log(3),... % Feature 4 log of base 3
      log(data(:,9))/log(3),... % Feature 9 log of base 3
      log2(data(:,11)),...      % Feature 11 log of base 2
      log(data(:,13))/log(4)    % Feature 13 log of base 4
      ];
  
[r c]=size(data);


% Normalize all the features
m=mean(data);
sigma=std(data);
dataN=(data-repmat(m,r,1))./repmat(sigma,r,1);


% Calculate predictions
x = [dataN, ones(size(dataN, 1), 1)];
yPred = abs((W'*x')');

csvwrite('prediction.csv', yPred);