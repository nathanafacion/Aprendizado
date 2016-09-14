# Declarar variavel 
x <-42
# Variavel True e False
y <- "Reduzido True(T) ou false(F)"
T
F
# Realizar soma
sum(1,3,5)
# Repeticao
rep("Yo ho!", times = 3)
# Raiz quadrada
sqrt(16)
# Usar o help: help(functionname)
# example(functionname)
# Lista artquivos
list.files()
# Codigo fonte
#source("bottle1.R")
# vetor , deve ser sempre do mesmo tipo
# senao sera convertido para string
c(4,5,7)
# sequencia
5:9
# sequencia
seq(5,9)
# sequencia o proximo numero da sequencia tera incremento 0.5
seq(5,9,0.5)
# sentence <- c('walk', 'the', 'plank')
# sentence[3]
# sentence[4]<- 'to' <- se nao existe a posicao no vetor ele adiciona
# senao ele coloco no lugar.
# sentence[c(1,3)] <- pega a primeira e terceira palavra do vetor
# sentence[2:4] <- pega do 2 ao 4 do vetor
# sentence[5:7] <- c('the', 'poop', 'deck')
# names(ranks) <- c("first", "second", "third") <- cria tres posicoes de vetor com este nome
# Plotar grafico de barras:
vesselsSunk <- c(4, 5, 1)
barplot(vesselsSunk)
# names(vesselsSunk) <- c("England", "France", "Norway") <- Essa atribuicao vai virar legenda no eixo x

# Operacoes matematicas
# a <- c(1, 2, 3)
# a + 1
# a/2
# Dois vetores iguais: a == c(1, 99, 3)  -> Resultado vai ser a comparacao de cada elemento se eh verdadeiro
# ou falso.
# funcao sin(x)  eh seno
# como dar plot no grafico
plot(x,y)
# funcao absoluta que desconsidera o sinal
abs(values)

# Valores NA sao desconsiderados ao realizar a soma
sum(a, na.rm = TRUE)

# Declaracao de Matrix
matrix(0,3,4)

# Criando matrix..
# dim recebe um vetor que vai popular uma matrix 2,4
dim(plank) <- c(2,4)
# print(variavel), printa na tela a variavel
# acessando valor da matrix
plank[2,3]
# imprime a linha toda
plank[2,]
# contorna o local dos valores da matrix
contour(elevation)
# coloca em uma perspectiva 3d
persp(elevation)
# O quanto voce quer diminuir a altura(eixo z)  do 3d que sera mostrado
persp(elevation, expand=0.2)
# mostrar a imagem
image(volcano)

# mostrar medias
mean(limbs)