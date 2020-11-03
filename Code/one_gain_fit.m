function error = one_gain_fit(b, x, y)
    % b = [gain, width, offset]
    % x = [oris, data/responses]
    % error = variance accounted for
    
    % w/ alpha (global gain):
%     y_fit = b(1)*(1-b(2)*exp(b(3)*cos(deg2rad(x(:,1))))).*x(:,2)+b(4);
    
    % w/o alpha
    y_fit = (1-b(1)*exp(b(2)*(cos(deg2rad(x(:,1)))-1))).*x(:,2)+b(3);
%     error = 1-mean((y_fit-y).^2)/var(y);
%     error = mean((y_fit-y).^2);
    error = sum((y_fit-y).^2);
end

% Anna's example:
% % create data
% b1 = 2;
% b2 = 3;
% b3 = 0.5;
% x = randn(1000,1);
% y = b1+b2*exp(x*b3)+randn(size(x))*0.5;
% 
% %arbitraty starting vals
% startvals = [1, 4, 0.2];
% 
% b = fminsearch(@(b) testfunc(b, x, y), startvals);
% 
% function error = testfunc(b, x, y)
% 
%     y_fit  = b(1)+b(2)*exp(x*b(3));
%     error = sum((y_fit-y).^2);
% end