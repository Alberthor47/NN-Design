% Transfer functions
function y = purelin(x)
    y = x; % Linear transfer function
end

function y = poslin(x)
    y = max(0, x); % Positive linear transfer function
end

function y = compet(x)
    % Compet transfer function, selects the highest element as 1, others as 0
    [~, idx] = max(x);
    y = zeros(size(x));
    y(idx) = 1;
end
