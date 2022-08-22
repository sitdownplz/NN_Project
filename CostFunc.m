classdef CostFunc
    % Cost function 
    properties
        NNmodel
    end
    
    methods
        function self = CostFunc
        end
    end

     methods (Static)
         
        % Sum square error
        function error = SSELoss(y, y_hat)
            error = sum((y - y_hat).^2)/2;
        end
        
        % Mean absolute error
        function error = MAELoss(y, y_hat)
            error = mean(abs(y - y_hat));
        end

    end
end
