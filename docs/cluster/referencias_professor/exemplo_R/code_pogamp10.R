#General code
rm(list=ls(all=TRUE))
library(MASS)
library(condMVNorm)
library(mvnfast)
library(sn)
library(actuar)
library(mnormt)
library(matrixcalc)

#####################
#Creating the S area#
#####################
xMin = 0
xMax=10
yMin=0
yMax=10
xDelta=xMax-xMin #rectangle dimensions
yDelta=yMax-yMin #rectangle dimensions
area=xDelta*yDelta

#####################
#Generating r points#
#####################
gridd1 = c(0.10, 1.08, 2.06, 3.04, 4.02, 5.00, 5.98, 6.96, 7.94, 8.92, 9.90) #fixed points for r
gridd2 = c(0.10, 1.08, 2.06, 3.04, 4.02, 5.00, 5.98, 6.96, 7.94, 8.92, 9.90) #fixed points for r
gridd = expand.grid(gridd1,gridd2)
gridd = as.matrix(gridd)

##################################################################
#Generating random variables from process Y for a fixed rectangle#
##################################################################
yprocess_article <- function(
    r,
    lambda.poi,
    xi = 0,
    tau = 1,
    delta = 0.99,
    tau2 = 10,
    l = 1.5
){
  
  # ----- Região -----
  xMin = 0; xMax = 10
  yMin = 0; yMax = 10
  area = (xMax-xMin)*(yMax-yMin)
  
  # ----- Processo de Poisson -----
  nN = rztpois(1, area * lambda.poi)
  
  xx = runif(nN, xMin, xMax)
  yy = runif(nN, yMin, yMax)
  Npts = cbind(xx, yy)
  
  # Junta r com N
  allpts = rbind(r, Npts)
  
  # Número total de pontos
  nTot = nrow(allpts)
  
  # ----- Matriz de covariância espacial Υ -----
  distAll = as.matrix(dist(allpts))
  Upsilon = exp(-(distAll^l)/tau2)
  
  # ----- Processo ε completo -----
  eps = as.numeric(
    mvrnorm(1, rep(0, nTot), Upsilon)
  )
  
  # ----- Processo Z (i.i.d half-normal) -----
  Z = abs(rnorm(nTot))
  
  # ----- Construção de Y -----
  Y = xi +
    tau * delta * Z +
    tau * sqrt(1 - delta^2) * eps
  
  # Retorna apenas o valor no ponto r
  return(Y[1])
}

m = 50000
r = c(5,5)

sim01  = replicate(m, yprocess_article(r, lambda.poi = .1))
sim02  = replicate(m, yprocess_article(r, lambda.poi = .2))
sim03 = replicate(m, yprocess_article(r, lambda.poi = .3))
sim005 = replicate(m, yprocess_article(r, lambda.poi = .05))

