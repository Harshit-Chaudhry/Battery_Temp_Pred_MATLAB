%% Load Data
clc; clear; close all;
filename = '/MATLAB Drive/Mya Datasets/3c 35du new.xlsx';
data = readtable(filename);
modelSavePath = '/MATLAB Drive/Mya Datasets/c3_35du_new_Models/';
imageSavePath = '/MATLAB Drive/Mya Datasets/c3_35du_new__Images/';
EPOCHS = 1;  % Increased epochs for better training
BATCH_SIZE = 32; % Mini-batch size
LEARNING_RATE = 0.001; % Lower learning rate for stable training

%% Create Directories if They Don't Exist
if ~exist(modelSavePath, 'dir')
    mkdir(modelSavePath);
end
if ~exist(imageSavePath, 'dir')
    mkdir(imageSavePath);
end

%% Prepare Data for Training
inputFeatures = data{:, {'Time', 'Current', 'AmbientTemp', 'Voltage', 'DiffTemp'}};
target = data{:, 'BatteryTemp'};

% Normalize data for better generalization
inputFeatures = normalize(inputFeatures);
target = normalize(target);

% Split Data (80% Training, 20% Testing)
splitPoint = round(0.8 * height(data));
XTrain = inputFeatures(1:splitPoint, :);
YTrain = target(1:splitPoint);
XTest = inputFeatures(splitPoint+1:end, :);
YTest = target(splitPoint+1:end);

%% Train Regression Tree Model
treeModel = fitrtree(XTrain, YTrain);
YPredTree = predict(treeModel, XTest);
rmseTree = sqrt(mean((YTest - YPredTree).^2));
disp(['Regression Tree RMSE: ' num2str(rmseTree)]);
save(fullfile(modelSavePath, 'regression_tree_model.mat'), 'treeModel');

%% Train Weighted Nearest Neighbors (WNN)
k = 5;
YPredKNN = zeros(size(XTest, 1), 1);
for i = 1:size(XTest, 1)
    testPoint = XTest(i, :);
    distances = sqrt(sum((XTrain - testPoint).^2, 2));
    [sortedDistances, indices] = sort(distances);
    nearestNeighbors = YTrain(indices(1:k));
    weights = 1 ./ (sortedDistances(1:k) + 1e-6);
    YPredKNN(i) = sum(nearestNeighbors .* weights) / sum(weights);
end
rmseKNN = sqrt(mean((YTest - YPredKNN).^2));
disp(['Manual WNN RMSE: ' num2str(rmseKNN)]);
save(fullfile(modelSavePath, 'manual_wnn_model.mat'), 'k', 'XTrain', 'YTrain');

%% Prepare Data for LSTM (Sequence Length = 10 for better learning)
sequenceLength = 10;
numSequences = height(data) - sequenceLength;
X = [];
Y = [];
for i = 1:numSequences
    X = cat(3, X, inputFeatures(i:i+sequenceLength-1, :)');
    Y = [Y; target(i+sequenceLength)];
end

% Train-Test Split for LSTM
XTrainLSTM = X(:, :, 1:round(0.8 * numSequences));
YTrainLSTM = Y(1:round(0.8 * numSequences));
XTestLSTM = X(:, :, round(0.8 * numSequences)+1:end);
YTestLSTM = Y(round(0.8 * numSequences)+1:end);

% Convert to Cell Format
XTrainCell = squeeze(num2cell(XTrainLSTM, [1 2]))';
XTestCell = squeeze(num2cell(XTestLSTM, [1 2]))';

%% Define Improved LSTM Model
numFeatures = size(inputFeatures, 2);
numHiddenUnits = 128; % Increased hidden units for better feature learning
numResponses = 1;

layers = [
    sequenceInputLayer(numFeatures, 'Normalization', 'zscore')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    dropoutLayer(0.2) % Dropout to reduce overfitting
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(numResponses)
    regressionLayer
];

%% Training Options with Early Stopping
options = trainingOptions('adam', ...
    'MaxEpochs', EPOCHS, ...
    'MiniBatchSize', BATCH_SIZE, ...
    'InitialLearnRate', LEARNING_RATE, ...
    'ValidationData', {XTestCell, YTestLSTM}, ...
    'ValidationFrequency', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch');

%% Train LSTM Network
net = trainNetwork(XTrainCell, YTrainLSTM, layers, options);

% Predict with LSTM
YPredLSTM = predict(net, XTestCell, 'SequenceLength', sequenceLength);
rmseLSTM = sqrt(mean((YTestLSTM - YPredLSTM).^2));
disp(['LSTM RMSE: ' num2str(rmseLSTM)]);
save(fullfile(modelSavePath, 'trained_LSTM_model.mat'), 'net');

%% Compare and Choose Best Model
models = {'Regression Tree', 'Manual WNN', 'LSTM'};
rmseValues = [rmseTree, rmseKNN, rmseLSTM];
[minRMSE, bestModelIdx] = min(rmseValues);
bestModel = models{bestModelIdx};

disp(['Best model: ' bestModel]);
disp(['Minimum RMSE: ' num2str(minRMSE)]);

% Save Comparison Results
save(fullfile(modelSavePath, 'model_comparison_results.mat'), 'rmseValues', 'bestModel', 'minRMSE');

%% Plot Results
figure;
plot(YTest, 'b-', 'LineWidth', 1.5); hold on;
plot(YPredTree, 'r--', 'LineWidth', 1.5);
plot(YPredKNN, 'g--', 'LineWidth', 1.5);
plot(YPredLSTM, 'm--', 'LineWidth', 1.5);
hold off;
legend('Actual', 'Regression Tree', 'Manual WNN', 'LSTM');
xlabel('Time Steps');
ylabel('Battery Temperature');
title('Model Comparison: Actual vs. Predicted');
saveas(gcf, fullfile(imageSavePath, 'model_comparison.png'));
