function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%


h = X * theta; % m*(n+1) ¡Á (n+1)* -> m*
J_part1 = sum((h-y).^2) /2 /m;

theta2 = theta(2:end, :);
J_part2 = sum(theta2.^2)/2/m*lambda;

J = J_part1 + J_part2;


% gradient
grad_ori = X' * (h-y) / m; %(n+1)*m ¡Á m*1 -> (n+1)*1
grad(1) = grad_ori(1);
grad(2:end) = grad_ori(2:end) + lambda/m*theta(2:end);






% =========================================================================

grad = grad(:);

end
