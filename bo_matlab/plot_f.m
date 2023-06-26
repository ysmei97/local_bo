%function [f] = f(y)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
y=0:0.001:1;
f1=8*cos(4*y.^0.5-0.4)-20*(y-0.6).^2+3+15*y;
f2=5*cos(20*(y.^1.5-0.8));
f=f1+2*f2;
%end

plot(y,f)
grid minor

