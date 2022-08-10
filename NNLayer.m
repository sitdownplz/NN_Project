classdef NNLayer < handle

    properties (Access=public)
        X {mustBeNumeric}    % Input data
        neuronNum {mustBeNumeric}    % Number of neurons
        W {mustBeNumeric}    % Weight
        b {mustBeNumeric}    % Bias
        actFcnMethod  {mustBeMember(actFcnMethod, {'tanh', 'sigmoid', 'none'})}= 'tanh'    % Activation funcion method
        actFcn = @tanh    % Activation function
    end

    methods
        function self = NNLayer(inputSize, outputSize, actFcnMethod)

            self.neuronNum = inputSize;

            if nargin < 2
                self.W = ones(inputSize, 1);
                self.b = zeros(inputSize, 1);
            elseif nargin > 2
                self.actFcnMethod = actFcnMethod;
                switch actFcnMethod
                    case 'tanh'
                        self.actFcn = @tanh;
                    case 'sigmoid'
                        self.actFcn = @(x) 1./(1+e.^x);
                    case 'none'
                        self.actFcn = @(x)x;
                end
                self.W = rand(outputSize, inputSize);
                self.b = rand(outputSize,1);
            else
                self.W = rand(outputSize, inputSize);
                self.b = rand(outputSize,1);
            end
        end

        function y = forward(self, x, varargin)
            self.X = x(:);
            if size(self.X,1) ~= self.neuronNum
                error('Input size must equal to neurons')
            end
            if isempty(varargin)
                y = self.actFcn(self.W * self.X + self.b);
            else
                switch inputname(2)
                    case 'W_temp'
                        y = self.actFcn(varargin{2} * self.X + self.b);
                    case 'b_temp'
                        y = self.actFcn(self.W * self.X + varagin);
                end
            end
        end



    end

end