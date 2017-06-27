gera<-function(n) -4*(1:n)**2 +5*(1:n) +rnorm(n,100,30)

y=gera(7)
x=1:7

f<-function(x,w) w[1]*x**5+w[2]*x**4+w[3]*x**3+w[4]*x**2+w[5]*x+w[6]

err<-function(w,lambda=0) sum((f(x,w)-y)**2)+lambda*sum(w**2)

res=optim(rep(1,6),err,method="L-BFGS-B")

g=function(x) f(x,res$par)

plot(y~x)
curve(g,from=1,to=7,col="red",add=T)

res=optim(rep(1,6),function(w) err(w,0.3),method="L-BFGS-B")
curve(g,from=1,to=7,col="blue",add=T)

res=optim((1,6),function(w) err(w,3),method="L-BFGS-B")