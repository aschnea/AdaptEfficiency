function error = two_gain_fit(b, i, x, y)
    % b = [global gain, n-s gain, n-s width, s-s gain, s-s width, offset]
    % x = [ori prefs, stimulus oris, responses]
    
    % w/ alpha (global gain):
%     y_fit = b(1)*(1-b(2)*exp(b(3)*cos(deg2rad(x(:,1))))).*(1-b(4)*exp(b(5)*cos(deg2rad(x(:,2))))).*x(:,3)+b(6);

    % without alpha
    y_fit = ((1-b(1)*exp(b(2)*(cos(deg2rad(i))-1)))'.*(1-b(3)*exp(b(4)*(cos(deg2rad(i))-1)))).*x+b(5);
%     error = 1-mean((y_fit-y).^2)/var(y);
%     error = mean((y_fit(:)-y(:)).^2);
    error = sum((y_fit(:)-y(:)).^2);
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