classdef NNLayer < handle
    % Create default neural network layer framework
    % Layer Object = NNLayer(input neuron number, output neuron number, activation funciton)
    properties (Access=public)
        X {mustBeNumeric}    % Input data
        neuronNum {mustBeNumeric}    % Number of neurons
        W {mustBeNumeric}    % Weight
        b {mustBeNumeric}    % Bias
        h {mustBeNumeric}    % Hidden State
        actFcnMethod {mustBeMember(actFcnMethod, {'tanh', 'sigmoid', 'none'})}= 'tanh'    % Activation funcion method
        actFcn = @tanh    % Activation function
    end

    methods
        function self = NNLayer(inputSize, outputSize, actFcnMethod)
            self.neuronNum = inputSize;
            % Currently, only allow single neural in the output layer
            if outputSize == 1  %TODO
                self.W = rand(1,inputSize);
                self.b = zeros(outputSize, 1);
                self.actFcnMethod = 'none';
                self.actFcn = @(x)x;
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

        % Forward function
        function y = forward(self, x, varargin)
            if size(x,1) ~= self.neuronNum
                error('Input size must equal to neurons')
            end
            if isempty(varargin)
                self.X = x;
                y = self.actFcn(self.W * self.X + self.b);
                self.h = self.W * self.X + self.b;
            else
                switch inputname(2)
                    case 'X_temp'   % Manul input X data
                        
                        y = self.actFcn(self.W * x + self.b);
                    case 'W_temp'   % Manul input weight data
                        y = self.actFcn(x * self.X + self.b);
                        self.h = self.W * self.X + self.b;
                    case 'b_temp'   % Manul input bias data
                        y = self.actFcn(self.W * self.X + x);
                        self.h = self.W * self.X + self.b;
                end
            end
        end



    end

end