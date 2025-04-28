function [rmse, stop_an, L, stop_rmse] = stopLength(A, y, x, events)
    Length = 1:1:90;
    stop_an = zeros(length(Length),1);
    rmse = zeros(length(Length),1);

    len = length(x);
    
    for i = 1:length(Length)
        Ltemp = Length(i);
        
        Xmat=zeros(len-Ltemp+1,Ltemp);
        for s=1:len-Ltemp+1
            Xmat(s,:)=x(s:s+Ltemp-1);
        end

        Ymat=Xmat.'*Xmat; % small LxL matrix
        [U,Sigma,V]=svd(Ymat); % only need eigenvec corr to smallest eigenvalue
        h=U(:,end);
        c1=[h(1); zeros(len-Ltemp,1)];
        r1=zeros(1,len);
        r1(1:Ltemp)=h.';
        Han=toeplitz(c1,r1);

        Aan = [A;Han];
        yan = [y; zeros(len-Ltemp+1,1)];

        xhat_an = (pinv(Aan)*yan).';
        
        rmse(i) = sqrt(mean((events'-xhat_an).^2));
        
        stop_an(i) = norm((y-A*xhat_an'),2)+ ...
                    norm((Han*xhat_an'),2) + 10*abs(Ltemp);

    end
    
    stops = [0:0.1:0.8,0.9:0.01:1];
    
    diff = stop_an(1)-min(stop_an);
    L = zeros(1, length(stops));
    stop_rmse = zeros(1, length(stops));
    
    for k = 1:length(stops)
        for i = 1:length(Length)
            if stop_an(i)-stop_an(1) + diff*stops(k) <=0
                L(k) = Length(i);
                stop_rmse(k) = rmse(i);
                break;
            end
        end
    end
    
    
    
end
        
    
    
