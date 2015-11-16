% Initialization
clear ; close all; clc
%% ======================= Part 1: Plotting =======================
fprintf('Plotting Data ...\n')
data = load('iris.txt');
%load input data
sepLen = data(:, 1);
sepWid = data(:, 2);
petLen = data(:, 3);
petWid = data(:, 4);
dat = [sepLen,sepWid,petLen,petWid];
%load output data
species = data(:, 5);
%numberfy output
% 0 = Iris-setosa
% 1 = Iris-versicolor
% 2 = Iris-virginica
m = length(species); % number of training examples
fprintf('Program paused. Press enter to continue.\n');
pause;
%% ======================= Part 2: Gradient Descent =======================
X = [ones(m, 1), dat]; % Add a column of ones to x
theta = ones(columns(X),1);
lambda = 0.1;
num_labels = 3;
[all_theta] = oneVSAll(X, species, num_labels, lambda);
fprintf('Program paused. Press enter to continue.\n');
pause;
%% ======================= Part 4: test the training =======================
pred = predictoneVSAll(all_theta, X);
fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == species)) * 100);
