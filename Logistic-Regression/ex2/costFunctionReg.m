function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
htheta=sigmoid(X*theta);
temp=theta;
r=theta.^2;
r(1)=theta(1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

x1=sum((-y'*log(htheta)-(1-y)'*log(1-htheta)));

J=(1/m)*x1+(lambda/(2*m))*(sum(r)-r(1));


















for  i=1:length(theta),
  if(i==1),
  temp(i)=(1/m)*sum(X(:,i)'*(htheta-y));
else
   temp(i)=(1/m)*sum(X(:,i)'*(htheta-y))+(lambda/m)*theta(i,1);
  end
  end

grad=temp;



% =============================================================

end
