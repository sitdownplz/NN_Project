classdef NNmodel < handle
    % Neural network object, contains whole network information
    properties
        LayerGraph
        Connection
        GradFcn
        X    % Predictor
        Y    % Response
        YPred    % Current predict result
        grad_W    % Current weight gradient of each layer 
        grad_b    % Current bias gradient of each layer
    end

    methods
        function self = NNmodel
        end

        % Evaluate network with predictor an d response
        function y = eval(self, x, y)
            if size(x,2)~= self.LayerGraph(1).neuronNum
                error('Data size not match with input layer')
            end
            self.X = x';
            self.Y = y;
            initialGain = 1/sqrt(100);
            input = self.X;
            for i = 1:length(self.LayerGraph)
                output = forward(self.LayerGraph(i),input);
                % Batch normalization 
                if i < 4    %TODO
                  output = normalize(output);
                end
                input = output;
            end
            y = output;
            self.YPred = y;
        end

        % Back Propagation for Roy's GD
        % Manual passing hidden state value between layer
        function y = backward(self, x, layer)
            X_temp = x;
            if layer < length(self.LayerGraph)
                for i = layer+1:length(self.LayerGraph)
                    output = forward(self.LayerGraph(i), X_temp, 'X_temp');
                    X_temp = output;
                end
            end
            y = X_temp;
        end

    end

end