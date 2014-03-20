clc; clear; close all

%% load training data
data = csvread('../handout/training.csv');
y=data(:,15);
X = data(:,1:14);


for i=1:size(X,2)/2
    figure(i);
    plot(log(X(:,i)), y, '.r');    
end
