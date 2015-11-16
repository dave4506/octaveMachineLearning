function [all_theta] = oneVSAll(X, y, num_labels, lambda)
  m = size(X, 1);
  n = size(X, 2);

  all_theta = zeros(num_labels, n);


  for c = 0:num_labels
    initial_theta = zeros(n, 1);
    options = optimset('GradObj', 'on', 'MaxIter', 50);
    all_theta((c+1),:) = fmincg (@(t)(costFunction(t, X, (y == c), lambda)),initial_theta, options);
  end
end
