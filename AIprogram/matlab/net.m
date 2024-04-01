clc
clear

lr=0.01
lmd=0
a0=(-1:0.05:1);
y=a0.^2 +rand(size(a0))*0.1;

W1=randn(10, 1);
W2=randn(1, 10);
% +norm(X,"fro")
subplot(121)
scatter(a0,y)
b1=1
b2=1
for i=1:2000
    
    z1=W1*a0+b1;
    a1=F(z1);
    z2=W2*a1+b2;
    a2=F(z2);
    subplot(122)
    plot(a0,a2)
%     L=(y-a2).^2 ;
    derivatives2=-2*(y-a2).*dF(z2);
    derivatives1=dF(z1).*(W2'*derivatives2);
    dL_dW2=derivatives2*a1';
    dL_dW1=derivatives1*a0';
    W2=W2-lr*dL_dW2-lr*lmd*W2;
    W1=W1-lr*dL_dW1-lr*lmd*W1;
    b1=b1-lr*derivatives1;
    b2=b2-lr*derivatives2;
    
    
end

function activation=F(x)
    activation=1./(1+exp(-x));
%     activation=max(0,x);
end

function derivatives=dF(x)
    derivatives=exp(-x)./(1+exp(-x)).^2;
%     derivatives=(x>0).*x;

end


