function [pars, log_L] = lda_pnf_fit(startvals,Data,lb,ub)

%  fit weibull function using maximum liklihood

%  Data for fit:
%      X Input : Xdata
%      Y Output: ans
%      lb = lower bound
%      ub = upper bound
%  Output:
%      fitresult : a fit object representing the fit.
%      gof : structure with goodness-of fit info.

% % %% Fit: 'untitled fit 1'.
% % [xData, yData] = prepareCurveData( Xdata, YData );
% % 
% % % Set up fittype and options.
% % ft=fittype('1-0.5*exp(-(c/a)^b)', 'independent', 'c', 'dependent', 'y' );
% % opts = fitoptions( 'Method', 'NonlinearLeastSquares' );
% % opts.Display = 'Off';
% % opts.StartPoint = [0 0 0];
% % % opts.StartPoint = [0.990124838277548 0.958378375329677 0.171741020420804];
% % % opts.Lower = [0 0 0]
% % 
% % % Fit model to data.
% % [fitresult, gof] = fit( xData, yData, ft, opts );


options = optimset('TolFun',1e-5,'TolX',1e-4,'Maxiter',10000,...
    'MaxFunEvals',10000,'Display','off','LargeScale','off');
[pars,log_L]=fmincon(@mymodel,startvals,[],[],[],[],lb,ub,[],options);

log_ub=sum(log(((Data.^Data).*exp(-1*Data))./factorial(round(Data))));
ave_resp=mean(Data)*ones(size(Data,1),size(Data,2));
log_lb=sum(log(((ave_resp.^Data).*exp(-1*ave_resp))./factorial(round(Data))));

log_L=[-1*log_L log_lb log_ub];
log_L=[log_L (log_L(1)-log_L(2))/(log_L(3)-log_L(2))];

    function [log_L] = mymodel(startvals)
%         xdata=2:2:2*length(Data); % for data
        xdata=1:1:length(Data); % for model
        a=startvals(1); % derivative in exponential: 'slope'
        b=startvals(2); % exponent in exponential: 'slope'
        c=startvals(3); % coefficient in front: vertically shifts intercept and stretches function
        predicted=1-c*exp(-(xdata/a).^b);
%         predicted=1-0.5*exp(-(xdata/a).^b);
        log_L=sum(log(((predicted.^Data).*exp(-1*predicted))./factorial(round(Data))));
        if isinf(log_L) || isnan(log_L)
            predicted=predicted/10;
            Data=Data/10;
            log_L=sum(log(((predicted.^Data).*exp(-1*predicted))./factorial(round(Data))));
        end
        log_L=-1*log_L;
        
    end
end
