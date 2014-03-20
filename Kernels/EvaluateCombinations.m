clear; close all

% Load the data
data = csvread('../handout/training.csv');

% Determine the baseline mean error
dataBaseline=[data(:,1),...         %
      data(:,2:5)/8,...     % Divide by increment
      data(:, 6)/2,...      % Divide by increment
      data(:,7:14),...      %
      data(:,3).^3,...      % Feature 3 by power of 3 (potentially also pw2)
      data(:,5).^3,...      % Feature 5 by power of 3
      data(:,12).^3,...     % Feature 12 by power of 3 
      data(:,13).^4,...     % Feature 13 by power of 4 (or pw3 or pw5)
      data(:,15)            % Labels for training data
      ];
[baselineMeanError, baselineErrorStd] = regressionfunc(20, dataBaseline);

% Evaluate all possible combinations of products
errors = zeros(4*14,5);
k = 1; %Run index
for i=2:5
    for j=1:14
        runData = [dataBaseline(:,1:(end-1)), log(data(:,j))/log(i), data(:,15)];
        [meanRunError, runErrorStd] = regressionfunc(20, runData);
        errors(k,:) = [k, i, j, meanRunError, runErrorStd];
        fprintf('Finished run %d \n', k);
        k = k + 1;
    end
end

baselineMeanError
baselineErrorStd

[er, ec] = size(errors);

errors(:,6) = errors(:,4) - repmat(baselineMeanError,er,1); % meanRunError - meanBaselineError

errors