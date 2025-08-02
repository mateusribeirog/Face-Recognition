clear; clc; close all;

pkg load statistics

D=load('coluna2.dat');

Nr=100;  % No. de repeticoes

Ptrain=80; % Porcentagem de treinamento

tic; [STATS_0 TX_OK0 X0 m0 S0 posto0]=mahalanobis(D,Nr,Ptrain); Tempo0=toc;    % One COV matrix per class

STATS_0

Tempo0

figure; boxplot(TX_OK0)

figure; histfit(TX_OK0)


