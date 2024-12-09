% FeedForwardLayer for Hamming Network
classdef feedForwardLayer
    properties
        Weights
        Bias
        TransferFunction
    end
    
    methods
        function obj = feedForwardLayer(W)
            % Constructor with prototypes as weights
            obj.Weights = W';
            s = size(W', 1);
            obj.Bias = ones(s, 1) * s; % Set bias as per size of weights
            obj.TransferFunction = @purelin; % Linear transfer function
        end
        
        function result = propagate(obj, inputObj)
            % Display metadata
            disp('TF in Feed Forward Layer:');
            disp(obj.TransferFunction);
            disp('W in Feed Forward Layer:');
            disp(obj.Weights);
            disp('b in Feed Forward Layer:');
            disp(obj.Bias);
            
            % Propagate through FeedForwardLayer
            result = obj.TransferFunction(obj.Weights * inputObj + obj.Bias);
            disp('Feed Forward Layer Result (a1):');
            disp(result);
        end
    end
end
