function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

C_array = [0.01;0.03;0.1;0.3;1;3;10;30];  %C数组
sigma_array = [0.01;0.03;0.1;0.3;1;3;10;30];  %sigma数组
error_array = zeros(8,8);  %创造错误率数组，第（i，j）个元素表示C（i），sigma(j)对应的错误率
error_min = 10000;  %记录当前最小错误率

for i = 1:8,
    for j = 1:8,
        model= svmTrain(X, y, C_array(i), @(x1, x2) gaussianKernel(x1, x2, sigma_array(j)));   %以C(i),sigma(j)为参数训练SVM，（X，y）为训练样本
        predictions = svmPredict(model, Xval);  %利用上面训练得到的model对交叉验证样本进行预测
        error_array(i,j) =  mean(double(predictions ~= yval));  %记录错误率
        if(error_array(i,j) < error_min)  %如果当前的错误率更小，则记录他，并记录此时的C和sigma
            error_min = error_array(i,j);
            C = C_array(i);
            sigma = sigma_array(j);
        end
    end
end


% =========================================================================

end
