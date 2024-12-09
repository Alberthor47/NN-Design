% Implementation of Hamming Network
classdef hammingNetwork
    properties
        feedForwardLayer
        recurrentLayer
    end
    
    methods
        function obj = hammingNetwork(prototypes)
            % Constructor
            obj.feedForwardLayer = feedForwardLayer(prototypes);
            obj.recurrentLayer = recurrentLayer();
        end
        
        function result = classify(obj, inputObj, aes)
            % Classify input object
            a1 = obj.feedForwardLayer.propagate(inputObj);
            recurrent_result = obj.recurrentLayer.propagate(a1, aes);
            result = compet(recurrent_result);
            disp('Hamming Network result:');
            disp(result);
        end
    end
end
