function [K1] = K1(x1,x2,theta0,theta1)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
l1=length(x1);
l2=length(x2);
x=x1*ones(1,l2)-ones(l1,1)*(x2');
K1=-1*(theta0/theta1^2)*x.*exp(-1*x.^2/(2*theta1^2));
end

