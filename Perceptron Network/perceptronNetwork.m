% Implementation of Perceptron Network (With learining)
classdef perceptronNetwork
    properties
        Weights
        Bias
        TransferFunction
    end
    
    methods
        function obj = perceptronNetwork(number_of_neurons, input_size)
            % Constructor
            obj.Weights = rand(number_of_neurons, input_size); % Build the Weight matrix, init with random values
            obj.Bias = rand(number_of_neurons, 1); % Build the Bias vector, init with random values
            obj.TransferFunction = @(x) hardlim(x); % TF in perceptron should always be hardlim (or hardlims)
        end
        
        function output = classify(obj, prototype)
            % Classify input using current weights and bias
            net_input = obj.Weights * prototype + obj.Bias;
            output = obj.TransferFunction(net_input);
        end
        
        function [obj, stop_code] = train(obj, prototypes, max_epochs)
            % Train perceptron using prototypes with a maximum epoch counter
            epoch = 0; % Initialize epoch counter
            stop_code = 1; % If trining is completed successfully stop code is 1
            
            while ~obj.correct(prototypes) && epoch < max_epochs
                epoch = epoch + 1; % Increment epoch counter
                for i = 1:length(prototypes)
                    input_v = prototypes{i}{1};
                    target = prototypes{i}{2};
                    
                    classification = obj.classify(input_v);
                    
                    % Update weights and bias
                    obj.Weights = obj.Weights + (target - classification) * input_v';
                    obj.Bias = obj.Bias + (target - classification);
                end
            end
            
            % Plot results based on mode
            if size(prototypes{1}{1}, 1) == 2
                obj.plot_2D(prototypes); % Plot decision boundary if input is 2D for clasification
            end

            % Check if training stopped due to reaching max epochs
            if epoch == max_epochs
                stop_code = -1; % Code is -1 when max epochs is stop reason
                fprintf('Training stopped: maximum epochs reached (%d).\n', max_epochs);
            else
                fprintf('Training completed successfully in %d epochs.\n', epoch);
            end
        end

        function all_targets_meet = correct(obj, prototypes)
            % Check if all classifications are correct
            all_targets_meet = true;
            for i = 1:length(prototypes)
                input_v = prototypes{i}{1};
                target = prototypes{i}{2};

                % Get the classification result
                classification = obj.classify(input_v);
                
                % Compare the classification with the target
                if ~isequal(target, classification)
                    all_targets_meet = false;
                    return; % Exit early
                end
            end
        end

        function plot_2D(obj, prototypes)
            % Initialize figure
            figure;
            hold on;

            % Add X and Y axis lines at zero
            line([-2.5, 2.5], [0, 0], 'Color', 'k', 'LineWidth', 1); % X-axis
            line([0, 0], [-2.5, 2.5], 'Color', 'k', 'LineWidth', 1); % Y-axis

            % Set axes limits
            xlim([-2.6, 2.6]);
            ylim([-2.6, 2.6]);
            
            % Labels and title
            title('Decision Boundary & Training Data Points Colored by Class');
        
            % Define colors for up to 4 classes
            colors = {'r', 'g', 'm', 'y'}; % Red, Green, Magenta, Yellow
        
            % Loop through prototypes and plot each point
            for i = 1:length(prototypes)
                input_v = prototypes{i}{1}; % Extract input vector
                target = prototypes{i}{2};   % Extract target class
                number_of_classes = size(target, 1) * 2;

                if (number_of_classes == 2) 
                    if isequal(target, 0)
                        class_index = 1; % Class 1
                    elseif isequal(target, 1)
                        class_index = 2; % Class 2
                    end

                    % Calculate coordinates for decision boundary line
                    x1_boundary = linspace(-2.6, 2.6, 100); % X values for plotting 
                    y1_boundary = - (obj.Weights(1) / obj.Weights(2)) * x1_boundary - (obj.Bias / obj.Weights(2)); % Y values from decision boundary equation

                    % Plot decision boundary line
                    plot(x1_boundary, y1_boundary, 'b--', 'LineWidth', 1.5); % Dashed black line for decision boundary
                    
                    % Draw the weight vector as an arrow (quiver)
                    origin = [0; 0]; % Origin point for the arrow (starting point)
                    quiver(origin(1), origin(2), obj.Weights(1), obj.Weights(2), 'k', 'LineWidth', 2, 'MaxHeadSize', 1); 
                    text(obj.Weights(1)/2, obj.Weights(2)/2, 'W', 'Color', 'k'); % Label for the weight vector

                elseif (number_of_classes == 4)
                    % Determine class based on binary vector
                    if isequal(target, [0; 0])
                        class_index = 1; % Class 1
                    elseif isequal(target, [1; 0])
                        class_index = 2; % Class 2
                    elseif isequal(target, [0; 1])
                        class_index = 3; % Class 3
                    elseif isequal(target, [1; 1])
                        class_index = 4; % Class 4
                    end

                    % Calculate coordinates for decision boundary line
                    x1_boundary = linspace(-2.6, 2.6, 100);
                    y1_boundary = - (obj.Weights(1) / obj.Weights(2)) * x1_boundary - (obj.Bias(2) / obj.Weights(2)); 
                    y2_boundary = - (obj.Weights(3) / obj.Weights(4)) * x1_boundary - (obj.Bias(1) / obj.Weights(4)); 

                    % Plot decision boundary line
                    plot(x1_boundary, y1_boundary, 'b-.', 'LineWidth', 1.5); 
                    plot(x1_boundary, y2_boundary, 'b--', 'LineWidth', 1.5); 
                    
                    % Draw the weight vector as an arrow (quiver)
                    origin = [0; 0]; % Origin point for the arrow (starting point)
                    quiver(origin(1), origin(2), obj.Weights(1), obj.Weights(2), 'k', 'LineWidth', 1.5, 'MaxHeadSize', 1); 
                    quiver(origin(1), origin(2), obj.Weights(3), obj.Weights(4), 'MarkerFaceColor', "#77AC30", 'LineWidth', 1.5, 'MaxHeadSize', 1); 
                end

                % Plot the point with the corresponding color
                scatter(input_v(1), input_v(2), 100, colors{class_index}, 'filled', ...
                            'DisplayName', sprintf('Class %d', class_index));  
            end

            hold off;
        end
    end
end
