% Implementation of ADALINE Network
classdef adalineNetwork
    properties
        Weights
        Bias
        TransferFunction
        Mode
    end
    
    methods
        function obj = adalineNetwork(input_size, mode)
            % Constructor
            obj.Weights = rand(1, input_size); % Initialize weights randomly
            obj.Bias = rand(1); % Initialize to 0 for simplicity
            obj.TransferFunction = @(x) purelin(x); % TF in adaline is linear
            % Validate and set the mode
            if nargin < 2
                mode = 'classification'; % Default mode is regression
            end
            validatestring(mode, {'classification', 'regression'}, 'adalineNetwork', 'mode');
            obj.Mode = mode;
        end
        
        function output = classify(obj, prototype)
            % Classify input using current weights and bias
            if strcmp(obj.Mode, 'classification')
                net_input = obj.Weights * prototype + obj.Bias;
                output = sign(net_input); % Classification outputs -1 or 1
            else
                net_input = obj.Weights * prototype;
                output = obj.TransferFunction(net_input); % Regression outputs raw linear value
            end
        end
        
        function [obj, stop_code] = train(obj, prototypes, max_epochs, delta, error_threshold)
            % Train ADALINE using prototypes with a maximum epoch counter and an error threshold.
            if nargin < 5 % Default value for error_threshold if not provided
                error_threshold = 1e-3; % Default threshold for MSE
            end
  
            epoch = 0; % Initialize epoch counter
            stop_code = 1; % If training is completed successfully stop code is 1
            epoch_errors = []; % Training epoch error history
            train_ready = 0;

            while ~train_ready && epoch < max_epochs
                epoch = epoch + 1; % Increment epoch counter
                total_error = 0; % Initialize total error for the epoch

                for i = 1:length(prototypes)
                    input_v = prototypes{i}{1}; % Extract input vector
                    target = prototypes{i}{2};   % Extract target value
                    
                    classification = obj.classify(input_v); % Get classification

                    % Calculate error
                    error = target - classification;
                    total_error = total_error + sum(error.^2);
                    
                    % Update weights and bias using the error
                    obj.Weights = obj.Weights + 2 * delta * error * input_v'; % Weight update
                    obj.Bias = obj.Bias + 2 * delta * error; % Bias update
                end

                % Compute Mean Square Error
                mse = total_error / length(prototypes);
               
                % Store mean squared error (MSE) for this epoch
                epoch_errors = [epoch_errors, mse];

                % Check if MSE is below the threshold
                train_ready = mse < error_threshold;
            end
            
            % Plot results based on mode
            if strcmp(obj.Mode, 'classification') && size(prototypes{1}{1}, 1) == 2
                obj.plot_2D_db(prototypes); % Plot decision boundary if input is 2D for clasification
            elseif strcmp(obj.Mode, 'regression')
                obj.plot_epoch_error(epoch_errors); % Plot error over epochs for regression
            end

            % Check if training stopped due to reaching max epochs
            if epoch == max_epochs
                stop_code = -1; % Code is -1 when max epochs is stop reason
                fprintf('Training stopped: maximum epochs reached (%d).\n', max_epochs);
            else
                fprintf('Training completed successfully in %d epochs.\n', epoch);
            end
        end
       
        function plot_2D_db(obj, prototypes)
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
                    if isequal(target, -1)
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
                    if isequal(target, [-1; -1])
                        class_index = 1; % Class 1
                    elseif isequal(target, [1; -1])
                        class_index = 2; % Class 2
                    elseif isequal(target, [-1; 1])
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
   
        function plot_epoch_error(obj, epoch_errors)
            figure;
            plot(1:length(epoch_errors), epoch_errors, 'b-', 'LineWidth', 2);
            xlabel('Epoch');
            ylabel('Mean Squared Error (MSE)');
            title('Training Error Over Epochs');
            grid on;
        end
            
    end
end
