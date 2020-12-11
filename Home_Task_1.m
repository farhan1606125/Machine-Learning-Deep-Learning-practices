clear all;
clc;
close all;
s=tf('s');
G=1971*(s+0.047)/(s*(s+29.1)*(s+0.01)*(s+10));
H=1;
T=feedback(G,H);
t=0:0.01:5;

step((1/s)*T,t);
in=t.*ones(1,length(t));
hold on
plot(t,in)