function [error] = Hcorrection(b, H, N)
    % equation taken from Vinje and Gallant 2002, equation 10
    % H = experimental entropy
    % N = # of trials used to calculate H
    H_fit = b(1)+b(2)/N+b(3)/N^2+b(4)/N^3;
    error = sum((H_fit-H).^2);
end