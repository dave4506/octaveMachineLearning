function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
alfa = X*theta;
bravo = -1*y'*log(sigmoid(alfa));
charlie = (1-y')*log(1-sigmoid(alfa));
J = (bravo - charlie)/m;

delta  = theta .^ 2;
delta([1],:) = [];

echo = ((lambda/(2*m))*sum(delta));
J = J + echo;


gradient = (sigmoid(alfa)-y);
grad = (gradient' * X)/m;
foxtrot = (lambda/m)*theta;
grad = grad + foxtrot';
grad(1) = grad(1) - foxtrot(1);

% =============================================================

end
