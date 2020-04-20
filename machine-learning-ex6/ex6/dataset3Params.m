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

C_array = [0.01;0.03;0.1;0.3;1;3;10;30];  %C����
sigma_array = [0.01;0.03;0.1;0.3;1;3;10;30];  %sigma����
error_array = zeros(8,8);  %������������飬�ڣ�i��j����Ԫ�ر�ʾC��i����sigma(j)��Ӧ�Ĵ�����
error_min = 10000;  %��¼��ǰ��С������

for i = 1:8,
    for j = 1:8,
        model= svmTrain(X, y, C_array(i), @(x1, x2) gaussianKernel(x1, x2, sigma_array(j)));   %��C(i),sigma(j)Ϊ����ѵ��SVM����X��y��Ϊѵ������
        predictions = svmPredict(model, Xval);  %��������ѵ���õ���model�Խ�����֤��������Ԥ��
        error_array(i,j) =  mean(double(predictions ~= yval));  %��¼������
        if(error_array(i,j) < error_min)  %�����ǰ�Ĵ����ʸ�С�����¼��������¼��ʱ��C��sigma
            error_min = error_array(i,j);
            C = C_array(i);
            sigma = sigma_array(j);
        end
    end
end


% =========================================================================

end
