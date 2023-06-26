clear

%f1=8*cos(4*y-0.4)-6*(y-0.6).^2+3+9*y;
%f2=5*cos(9*(y-0.8));
%f=f1+2*f2;
theta0=10;
theta1=0.1;
T=12;
epsilon=0.1;

%sampled x
xx=(0:0.01:1)';

%x_star=rand(3,1);
V=[];
x_star=[0.25;0.5;0.75];
for k=1:19
    for i=1:1:length(xx)
        x=xx(i);
        KK=[K(x_star,x_star,theta0,theta1)+eye(length(x_star))*1e-6]^(-1);
        temp0 = K(x_star,x_star,theta0,theta1);
        temp1 = K(x,x_star,theta0,theta1);
        temp2 = K1(x,x_star,theta0,theta1);
        temp3 = f(x_star);
        mu=[K(x,x_star,theta0,theta1);K1(x,x_star,theta0,theta1)]*KK*f(x_star);
        nu=[K(x,x,theta0,theta1), 0; 0, K2(x,x,theta0,theta1)];
        aa=[K(x,x_star,theta0,theta1);K1(x,x_star,theta0,theta1)];
        nu=nu-aa*KK*aa';
        nu=nu+1e-6*eye(2);
        % test if assuming independent
        %nu(1,2)=0;
        % estimate the probability
        a2=( qfunc((-1*epsilon-mu(2))/sqrt(nu(2,2)))-qfunc((1*epsilon-mu(2))/sqrt(nu(2,2))) ) / (2*epsilon);
        a11= qfunc( (T-mu(1)+nu(1,2)*(0-mu(2))/nu(2,2)) / sqrt(nu(1,1)-nu(1,2)^2/nu(2,2)) );
        inte=0;
        for z=T:0.2:T+30
            inte=inte+0.2*qfunc( (z-mu(1)+nu(1,2)*(0-mu(2))/nu(2,2)) / sqrt(nu(1,1)-nu(1,2)^2/nu(2,2)) );
        end
        a1=inte;
        % expect improvement over T   \int_{-e<f'<e} E[ (f-T)*1(f>T) | f'] * P(f') 
        %A(i)= a1*a2;
        % probability larger than T   log  P(f>T | f') + a_t*log P(f')
        A(i)= a1*a2;
        % Add uncertainty
        %A(i)= a1*a2 + 1e-8*sqrt(nu(1,1));
        %exclude any points already sampled
        if min(abs(x-x_star))<0.011
            A(i)=0;
        end
    end

    
    %choose new sample point
    [v,j]=max(A);
    x_new = (j-1)/(length(xx)-1);
    V=[V, v];
    
    % plot
    y=0:0.001:1;
    subplot(2,1,1)
    hold on
    grid on
    if k==1
        plot(y,f(y))
        plot(x_star,f(x_star),'r*')
    end
    plot(x_new,f(x_new),'go')
    str={'',num2str(k)};
    text(x_new,f(x_new),str,'Color','green','FontSize',10)
    hold off
    


    subplot(2,1,2)
    %hold on
    plot(xx',A)
    grid off
    grid minor
    %hold off
    % update
    x_star = [x_star; x_new];
end
