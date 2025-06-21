%% 1. Load Data
filename = '/MATLAB Drive/Main_Mya/2c 15du new.xlsx';
data = readtable(filename);
modelSavePath = '/MATLAB Drive/Main_Mya/c2_15du_new_Models/';
imageSavePath = '/MATLAB Drive/Main_Mya/c2_15du_new_Images/';

% Training Parameters
EPOCHS = 100;
batchSize = 32;
initialLearnRate = 0.001;
sequenceLength = 20; 

% Create Directories If They Don't Exist
if ~exist(modelSavePath, 'dir')
    mkdir(modelSavePath);
end
if ~exist(imageSavePath, 'dir')
    mkdir(imageSavePath);
end

%% 2. Prepare Data
inputFeatures = data{:, {'Time', 'Current', 'AmbientTemp', 'Voltage', 'DiffTemp'}};
target = data{:, 'BatteryTemp'};
inputFeatures = normalize(inputFeatures);

% Split into Train and Test
splitPoint = round(0.8 * height(data));
XTrain = inputFeatures(1:splitPoint, :);
YTrain = target(1:splitPoint);
XTest = inputFeatures(splitPoint+1:end, :);
YTest = target(splitPoint+1:end);

%% 3. Regression Tree Model
treeModel = fitrtree(XTrain, YTrain);
YPredTree = predict(treeModel, XTest);
rmseTree = sqrt(mean((YTest - YPredTree).^2));
disp(['Regression Tree RMSE: ' num2str(rmseTree)]);
save(fullfile(modelSavePath, 'regression_tree_model.mat'), 'treeModel');

%% 4. Manual Weighted Nearest Neighbors (WNN)
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

%% 5. Prepare Data for LSTM/GRU
numSequences = height(data) - sequenceLength;
X = [];
Y = [];

for i = 1:numSequences
    X = cat(3, X, inputFeatures(i:i+sequenceLength-1, :)');
    Y = [Y; target(i+sequenceLength)];
end

% Split into training and testing
XTrainLSTM = X(:, :, 1:round(0.8 * numSequences));
YTrainLSTM = Y(1:round(0.8 * numSequences));
XTestLSTM = X(:, :, round(0.8 * numSequences)+1:end);
YTestLSTM = Y(round(0.8 * numSequences)+1:end);

XTrainCell = squeeze(num2cell(XTrainLSTM, [1 2]))';
XTestCell = squeeze(num2cell(XTestLSTM, [1 2]))';

numFeatures = size(inputFeatures, 2);
numHiddenUnits = 200;
numResponses = 1;

%% 6. 2-Layer LSTM Model
layersLSTM = [
    sequenceInputLayer(numFeatures, 'Normalization', 'zscore')
    lstmLayer(200, 'OutputMode', 'sequence')
    dropoutLayer(0.2)
    lstmLayer(100, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numResponses)
    regressionLayer
];

optionsLSTM = trainingOptions('adam', ...
    'MaxEpochs', EPOCHS, ...
    'MiniBatchSize', batchSize, ...
    'InitialLearnRate', initialLearnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'ValidationData', {XTestCell, YTestLSTM}, ...
    'ValidationPatience', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

netLSTM = trainNetwork(XTrainCell, YTrainLSTM, layersLSTM, optionsLSTM);

% LSTM Prediction
YPredLSTM = predict(netLSTM, XTestCell);
rmseLSTM = sqrt(mean((YTestLSTM - YPredLSTM).^2));
disp(['2-Layer LSTM RMSE: ' num2str(rmseLSTM)]);
save(fullfile(modelSavePath, 'trained_LSTM_model_2layer.mat'), 'netLSTM');

%% 7. 2-Layer GRU Model
layersGRU = [
    sequenceInputLayer(numFeatures, 'Normalization', 'zscore')
    gruLayer(200, 'OutputMode', 'sequence')
    dropoutLayer(0.2)
    gruLayer(100, 'OutputMode', 'last')
    dropoutLayer(0.2)
    fullyConnectedLayer(numResponses)
    regressionLayer
];

optionsGRU = trainingOptions('adam', ...
    'MaxEpochs', EPOCHS, ...
    'MiniBatchSize', batchSize, ...
    'InitialLearnRate', initialLearnRate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.5, ...
    'LearnRateDropPeriod', 10, ...
    'ValidationData', {XTestCell, YTestLSTM}, ...
    'ValidationPatience', 10, ...
    'Plots', 'training-progress', ...
    'Verbose', false);

netGRU = trainNetwork(XTrainCell, YTrainLSTM, layersGRU, optionsGRU);

% GRU Prediction
YPredGRU = predict(netGRU, XTestCell);
rmseGRU = sqrt(mean((YTestLSTM - YPredGRU).^2));
disp(['2-Layer GRU RMSE: ' num2str(rmseGRU)]);
save(fullfile(modelSavePath, 'trained_GRU_model_2layer.mat'), 'netGRU');

%% 8. Compare Models
models = {'Regression Tree', 'Manual WNN', '2-Layer LSTM', '2-Layer GRU'};
rmseValues = [rmseTree, rmseKNN, rmseLSTM, rmseGRU];
[minRMSE, bestModelIdx] = min(rmseValues);
bestModel = models{bestModelIdx};

disp(['Best model: ' bestModel]);
disp(['Minimum RMSE: ' num2str(minRMSE)]);
save(fullfile(modelSavePath, 'model_comparison_results.mat'), 'rmseValues', 'bestModel', 'minRMSE');

%% 9. Save Comparison Plot
figure;
plot(YTest, 'b-', 'LineWidth', 1.5);
hold on;
plot(YPredTree, 'r--', 'LineWidth', 1.5);
plot(YPredKNN, 'g--', 'LineWidth', 1.5);
plot(YPredLSTM, 'm--', 'LineWidth', 1.5);
plot(YPredGRU, 'c--', 'LineWidth', 1.5);
hold off;

legend('Actual', 'Regression Tree', 'Manual WNN', '2-Layer LSTM', '2-Layer GRU');
xlabel('Time Steps');
ylabel('Battery Temperature');
title('Model Comparison: Actual vs. Predicted');
grid on;
saveas(gcf, fullfile(imageSavePath, 'model_comparison_2layer.png'));
