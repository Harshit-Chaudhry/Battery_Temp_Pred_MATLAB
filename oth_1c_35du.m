clc; clear;

%% --- 1. Define Save Paths ---
modelSavePath = '/MATLAB Drive/Battery_Prediction_Models/c1_15du_newoth_Models/';
imageSavePath = '/MATLAB Drive/Battery_Prediction_Models/c1_15du_newoth_Images/';

% ✅ Create directories before saving
if ~exist(modelSavePath, 'dir'); mkdir(modelSavePath); end
if ~exist(imageSavePath, 'dir'); mkdir(imageSavePath); end

%% --- 2. Load Data ---
filename = '/MATLAB Drive/Battery_Prediction_Models/1c 15du new.xlsx';
data = readtable(filename);


%% --- 3. Prepare Features and Target ---
inputFeatures = data{:, {'Current', 'AmbientTemp', 'Voltage', 'Time'}};
target = data{:, 'BatteryTemp'};

splitPoint = round(0.8 * height(data));
XTrain = inputFeatures(1:splitPoint, :);
YTrain = target(1:splitPoint);
XTest = inputFeatures(splitPoint+1:end, :);
YTest = target(splitPoint+1:end);

%% --- 4. Random Forest ---
rfModel = fitrensemble(XTrain, YTrain, 'Method', 'Bag');
YPredRF = predict(rfModel, XTest);
rmseRF = sqrt(mean((YTest - YPredRF).^2));
disp(['Random Forest RMSE: ' num2str(rmseRF)]);

%% --- 5. Linear Regression ---
linearModel = fitlm(XTrain, YTrain);
YPredLinear = predict(linearModel, XTest);
rmseLinear = sqrt(mean((YTest - YPredLinear).^2));
disp(['Linear Regression RMSE: ' num2str(rmseLinear)]);

%% --- 6. SVM Regression ---
svmModel = fitrsvm(XTrain, YTrain, 'KernelFunction', 'gaussian');
YPredSVM = predict(svmModel, XTest);
rmseSVM = sqrt(mean((YTest - YPredSVM).^2));
disp(['SVM Regression RMSE: ' num2str(rmseSVM)]);



%% --- 7. Summary ---
models = {'Random Forest', 'Linear Regression', 'SVM', };
rmseValues = [rmseRF, rmseLinear, rmseSVM];
[minRMSE, idx] = min(rmseValues);
disp(['Best Model: ' models{idx}]);
disp(['Minimum RMSE: ' num2str(minRMSE)]);

%% --- 8. Plot Results ---
figure;
plot(YTest, 'k', 'LineWidth', 1.5); hold on;
plot(YPredRF, '--r', 'LineWidth', 1.2);
plot(YPredLinear, '--b', 'LineWidth', 1.2);
plot(YPredSVM, '--y', 'LineWidth', 1.2);

legend('Actual', 'Random Forest', 'Linear', 'SVM');
xlabel('Test Sample Index');
ylabel('Battery Temperature');
title('Model Comparison');
grid on;

drawnow;

saveas(gcf, fullfile(imageSavePath, 'ModelComparison.png'));

%% --- 10. Save Models ---
save(fullfile(modelSavePath, 'rfModel.mat'), 'rfModel');
save(fullfile(modelSavePath, 'linearModel.mat'), 'linearModel');
save(fullfile(modelSavePath, 'svmModel.mat'), 'svmModel');
disp('Models saved successfully.');