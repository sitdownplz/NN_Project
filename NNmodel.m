classdef NNmodel 
    properties
        LayerGraph
        Connection
        GradFcn
    end

    methods
        function self = NNmodel
        end
        function y = eval(self, x)
            if size(x,2)~= self.LayerGraph(1).neuronNum
                error('Data size not match with input layer')
            end
            initialGain = 1/sqrt(numel(x));
            input =x;
            for i = 1:length(self.LayerGraph)
                if i <length(self.LayerGraph)
                    self.LayerGraph(i).W = self.LayerGraph(i).W*initialGain;
                end
                output = forward(self.LayerGraph(i),input);
                input = output;
            end
            y = output;

        end

    end

end