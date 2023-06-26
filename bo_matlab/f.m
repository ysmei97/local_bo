function [f] = f(y)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
%f1=8*cos(4*y-0.4)-6*(y-0.6).^2+3+9*y;
%f2=5*cos(9*(y-0.8));
%f=f1+2*f2;

f1=8*cos(4*y.^0.7-0.4)-20*(y-0.6).^2+25*(y)+y.^2;
f2=5*cos(20*((y).^2.2-0.8));
f=f1+2*f2;
end


