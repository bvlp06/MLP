clear;
clc;
close all;
%%%%%%%%%%%%%%%Variable Declaration%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=6; %no of inputs
r=1; %no of outputs
m=8; %no of neurons in 1st hidden layer
l=4; %no of neurons in 2nd hidden layer
gamma_g=0.1;
gamma_m=0.5;
b=1;
w=0.2*rand(m,n)-0.1; %weights of 1st hidden layer
v=0.2*rand(l,m)-0.1; %weights of 2nd hidden layer
u=0.2*rand(r,l)-0.1; %weights of output layer
delw=zeros(m,n);
delv=zeros(l,m);
delu=zeros(r,l);
bolsaIn=load('B:\engComp2016\Disciplinas\8p TCC Monografia - Prof Marta de Oliveira Barreiros\rna\LAME4\LAME4.txt');
maior=max(bolsaIn(:,1));
menor=min(bolsaIn(:,1));
tam=size(bolsaIn);
bolsa=(bolsaIn-min(bolsaIn))./(max(bolsaIn)-min(bolsaIn));
inp=bolsa(:,1);
x=1:1:tam(1);
y_out=zeros(1,tam(1));
%%%%%%%%%%%%%%%Training%%%%%%%%%%%%%%%%%%%%%%%%%%%%
limit = round(tam(1)*0.8);
for epoch=1:9000
 for t = 1:limit
 gamma_m=gamma_m*t/limit;
 
 x2=bolsa(t,2);
 x3=bolsa(t,3);
 x4=bolsa(t,4);
 x5=bolsa(t,5);
 x6=bolsa(t,6);
 
 vec_x=[x2;x3;x4;x5;x6;b];
 a=w*vec_x;
 c=1./(1+exp(-a));
 d=v*c;
 e=1./(1+exp(-d));
 y_out(t)=u*e;
 y_des=inp(t);
 ey=y_des-y_out(t);
 ee=u' * ey;
 ed=e.*(1-e).*ee;
 ec=v' * ed;
 ea=c.*(1-c).*ec;
 error(t)=ey;
 delu=gamma_g*ey.*e' + gamma_m*delu;
 delv=gamma_g*ed.*c' + gamma_m*delv;
 delw=gamma_g*ea.*vec_x' + gamma_m*delw;
 u=u+delu;
 v=v+delv;
 w=w+delw;
 end
 train_mse(epoch) = mean((error).^2);
end
%%%%%%%%%%%%%%%Testing%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for t = limit+1:tam(1)

 z2=bolsa(t,2);
 z3=bolsa(t,3);
 z4=bolsa(t,4);
 z5=bolsa(t,5);
 z6=bolsa(t,6);
 vec_z=[z2;z3;z4;z5;z6;b];
 a1=w*vec_z;
 c1=1./(1+exp(-a1));
 d1=v*c1;
 e1=1./(1+exp(-d1));
 y_out(t)=u*e1;
 y_test(t)=u*e1;
 err(t)=inp(t)-y_test(t);
end
test_mse=mean((err).^2);
%%%%%%%%%%%%%%%Output Plotting%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%difference=inp(limit+1:2494)-y_test(limit+1:2494);

%figure;
%plot(x,inp,'r-',x,y_out,'b:',x,y_test,'g:','lineWidth',2)
%axis([0 tam(1) 0 1]);
%grid on
%xlabel('Quantidade de registros no dataset')
%ylabel('Saída da rede neural')
%legend('Saída desejada','Saída de treinamento','Saída de teste','Localização','Norte')
%title('Rede MLP - Predição da ação OIBR3 ')
%figure;
%plot(train_mse,'b');
%grid on
%xlabel('nº de épocas')
%ylabel('erro do treino - MSE')
%title('Erro durante o treinamento - OIBR3')
fprintf(1,'mse test error is %f',test_mse)

%difference=inp(limit+1:2494)-y_test(limit+1:2494);
inpdenormalized = ((y_test(limit+1:tam(1)))* (maior - menor) + menor);

bb = bolsaIn(limit+1:tam(1),1);

difference=bb-inpdenormalized.';
tabela=cell((tam(1)-limit+1),3);
    tabela(1,1)={'Fechamento Real'};
    tabela(1,2)={'Fechamento Predito'};
    tabela(1,3)={'Diferença'};
    tabela(2:end,1)= num2cell(bb);
    tabela(2:end,2)= num2cell(inpdenormalized);
    tabela(2:end,3)= num2cell(difference);

p=1:1:(tam(1)-limit); 
b5=0;

for i=1:size(p,2)
   if((difference(i,1))<0)
       difference(i,1)=difference(i,1)*(-1);
   end
   if ((difference(i,1))<=0.5)
       b5=b5+1;
   end    
end

figure
plot(p,difference.','k--o')
axis([1 size(p,2) round(min(difference)) ceil(max(difference))]);
grid on
xlabel('Nº de registros preditos')
ylabel('Diferença entre o valor real e o pretido')
title('Diferença - ação VALE4')
aa='VALE4';

%disp(limit);
%disp(size(p,2));
%disp(b5);

%disp(b5/(size(p,2))*100);
%disp(min(difference));
%disp(max(difference));
disp('\n /n')
dd=[' Na Figura 34 está apresentado os resultados da previsão da ação ',aa,' onde ',num2str(limit),' registros correspondem aos 80% do treinamento da rede. Ao desnormalizar os dados de saída da rede, foi calculado a diferença dos valores desejados com os valores preditos da rede. \n A maior diferença para ação ',aa,' foi de ',num2str(max(difference)),' dólares, e a menor diferença foi de ',num2str(min(difference)),' centavos. A quantidade de valores preditos correspondente aos 20% do dataset foi de ',num2str(size(p,2)),' sendo que ',num2str(b5),' ficaram dentro do limite estabelecido de meio dolar, ou seja, cerca de ',num2str((b5/(size(p,2))*100)),'% como mostra a Figura xx.'];
disp(dd)