%% Initialization
clear ; close all; clc

%% Load Data
%  The first x columns would contain the paramater value replace 2 below with the number of parameters
%  and 3 with the column with label(normal or abnormal)

data = load('data.txt');
X = data(:, [1, 2]); y = data(:, 3);



%% ============  Compute Cost and Gradient ============


%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);

% Add intercept term to x and X_test
X = [ones(m, 1) X];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============= Part 3: Optimizing using fminunc  =============


%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);



fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============== Part 4: Predict==============
% here we can change the values below for prediction

prob = sigmoid([1 45 85] * theta);
fprintf(['For scores 45 and 85, we predict an normality ' ...
         'probability of %f\n\n'], prob);

% Compute accuracy on our training set
p = predict(theta, X);


fprintf('\nProgram paused. Press enter to continue.\n');
pause;