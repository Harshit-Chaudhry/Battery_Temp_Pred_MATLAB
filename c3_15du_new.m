%% 1. Load Data
clc; clear; close all;
filename = '/MATLAB Drive/Mya Datasets/3c 15du new.xlsx';
data = readtable(filename);
modelSavePath = '/MATLAB Drive/Mya Datasets/c3_15du_new_Models/';
imageSavePath = '/MATLAB Drive/Mya Datasets/c3_15du_new_Images/';
EPOCHS = 2;
BATCH_SIZE = 32;
LEARNING_RATE = 0.001;

%% Create Directories if They Don't Exist
if ~exist(modelSavePath, 'dir')
    mkdir(modelSavePath);
end
if ~exist(imageSavePath, 'dir')
    mkdir(imageSavePath);
end

%% 2. Prepare Data
inputFeatures = data{:, {'Time', 'Current', 'AmbientTemp', 'Voltage', 'DiffTemp'}};
inputFeatures = normalize(inputFeatures);
target = normalize(data{:, 'BatteryTemp'});

% Split Data (80% Training, 20% Testing)
splitPoint = round(0.8 * height(data));
XTrain = inputFeatures(1:splitPoint, :);
YTrain = target(1:splitPoint);
XTest = inputFeatures(splitPoint+1:end, :);
YTest = target(splitPoint+1:end);

%% 3. Regression Tree Model
treeModel = fitrtree(XTrain, YTrain);
YPredTree = predict(treeModel, XTest);
rmseTree = sqrt(mean((YTest - YPredTree).^2));
save(fullfile(modelSavePath, 'regression_tree_model.mat'), 'treeModel');

%% 4. Weighted Nearest Neighbors (WNN)
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
save(fullfile(modelSavePath, 'manual_wnn_model.mat'), 'k', 'XTrain', 'YTrain');

%% 5. Prepare Data for LSTM
sequenceLength = 10;
numSequences = height(data) - sequenceLength;
X = [];
Y = [];
for i = 1:numSequences
    X = cat(3, X, inputFeatures(i:i+sequenceLength-1, :)');
    Y = [Y; target(i+sequenceLength)];
end
XTrainLSTM = X(:, :, 1:round(0.8 * numSequences));
YTrainLSTM = Y(1:round(0.8 * numSequences));
XTestLSTM = X(:, :, round(0.8 * numSequences)+1:end);
YTestLSTM = Y(round(0.8 * numSequences)+1:end);
XTrainCell = squeeze(num2cell(XTrainLSTM, [1 2]))';
XTestCell = squeeze(num2cell(XTestLSTM, [1 2]))';

%% 6. LSTM Model with Regularization
numFeatures = size(inputFeatures, 2);
numHiddenUnits = 128;
numResponses = 1;

layers = [
    sequenceInputLayer(numFeatures, 'Normalization', 'zscore')
    lstmLayer(numHiddenUnits, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(64)
    reluLayer
    fullyConnectedLayer(numResponses)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', EPOCHS, ...
    'MiniBatchSize', BATCH_SIZE, ...
    'InitialLearnRate', LEARNING_RATE, ...
    'ValidationData', {XTestCell, YTestLSTM}, ...
    'ValidationFrequency', 5, ...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'Shuffle', 'every-epoch');

net = trainNetwork(XTrainCell, YTrainLSTM, layers, options);
YPredLSTM = predict(net, XTestCell, 'SequenceLength', sequenceLength);
rmseLSTM = sqrt(mean((YTestLSTM - YPredLSTM).^2));
save(fullfile(modelSavePath, 'trained_LSTM_model.mat'), 'net');

%% 7. Compare Models
models = {'Regression Tree', 'Manual WNN', 'LSTM'};
rmseValues = [rmseTree, rmseKNN, rmseLSTM];
[minRMSE, bestModelIdx] = min(rmseValues);
bestModel = models{bestModelIdx};
save(fullfile(modelSavePath, 'model_comparison_results.mat'), 'rmseValues', 'bestModel', 'minRMSE');

%% 8. Save and Plot Results
figure;
plot(YTest, 'b-', 'LineWidth', 1.5);
hold on;
plot(YPredTree, 'r--', 'LineWidth', 1.5);
plot(YPredKNN, 'g--', 'LineWidth', 1.5);
plot(YPredLSTM, 'm--', 'LineWidth', 1.5);
hold off;
legend('Actual', 'Regression Tree', 'Manual WNN', 'LSTM');
xlabel('Time Steps');
ylabel('Battery Temperature');
title('Model Comparison: Actual vs. Predicted');
saveas(gcf, fullfile(imageSavePath, 'model_comparison.png'));
