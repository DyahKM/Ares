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
        
        tempL = L_stop(ratio);
        
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
    
    
