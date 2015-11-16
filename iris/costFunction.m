%% ======================= costFunction ==================================

% returns the cost function of the function and the gradientDescent needed
function [J,grad] = costFunction(theta, X, y, lambda)

  m = length(y);
  J= 0;
  grad = zeros(size(theta));
  h = sigmoid(X*theta);
  bravo = -1*y'*log(h);
  charlie = (1-y')*log(1-h);
  J = (bravo - charlie)/m;

  gradient = (h-y);
  grad = (gradient' * X)/m;
  foxtrot = (lambda/m)*theta;
  grad = grad + foxtrot';
  grad(1) = grad(1) - foxtrot(1);
  grad = grad(:);

end
