function [Out_A] = annihilating(Out, events, ratio)
    
    len_Out = length(Out);
	
    for l = 1:len_Out
        reconX = Out(l).x_reconstr;
        A = Out(l).A;
        y = Out(l).y;
        
        [rmse, stop_an, L_stop, stop_rmse] = stopLength(A, y, reconX, events);
        Out_A(l).rmse = rmse;
        Out_A(l).stop_an = stop_an;
        Out_A(l).L_stop = L_stop;
        Out_A(l).stop_rmse = stop_rmse;
        
        Out_A(l).A = A;
        Out_A(l).y = y;
        
        if isnumeric(ratio) && ratio > 0 && ratio <= length(L_stop)
            tempL = L_stop(ratio);
        elseif isnumeric(ratio) && (ratio > 0 && ratio <= 1)
            % Assuming L_stop corresponds to values between 0-1
            % Find closest match
            stops = [0:0.1:0.8,0.9:0.01:1]; % Same as in stopLength.m
            [~, idx] = min(abs(stops - ratio));
            tempL = L_stop(idx);
        else
            % Default to middle value if out of range
            tempL = L_stop(ceil(length(L_stop)/2));
            fprintf('Warning: Invalid ratio value, using default filter length.\n');
        end
        
        N = length(reconX);

        reconX = reconX.';
        Xmat=zeros(N-tempL+1,tempL);

        for s=1:N-tempL+1
            Xmat(s,:)=reconX(s:s+tempL-1);
        end

        Ymat=Xmat.'*Xmat; % small LxL matrix
        [U,Sigma,V]=svdecon(Ymat); % only need eigenvec corr to smallest eigenvalue, but since only LxL matrix, use full SVD here
        h=U(:,end);
        c1=[h(1); zeros(N-tempL,1)];
        r1=zeros(1,N);
        r1(1:tempL)=h.';
        H=toeplitz(c1,r1);

        xhat = (pinv([A; H])*[y; zeros(N-tempL+1,1)]).';
        error = sqrt(mean((xhat' - events).^2,1));

        
        Out_A(l).muvars = Out(l).muvars;
        Out_A(l).Matrix = H;
        Out_A(l).x_reconstr = xhat';
        Out_A(l).error = error;
        Out_A(l).L = tempL;
        Out_A(l).h = h;
        
    end
end
    
    
