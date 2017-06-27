#ostra a ideia de criar um grafico com linhas
pounds <- c(45000, 50000, 35000, 40000, 35000, 45000, 10000, 15000)
barplot(pounds)
meanValue <- mean(pounds)
deviation <-sd(pounds)
#  necesssario plotar o grafico antes de usar essas propriedades
abline(h = meanValue + deviation)
abline(h = meanValue)


# aplicando factor
chests <- c('gold', 'silver', 'gems', 'gold', 'gems')
types <- factor(chests)
print(chests)
print(types)
levels(types) # vai ter apenas os nao repetidos

#plota grafico com factor
weights <- c(300, 200, 100, 250, 150)
prices <- c(9000, 5000, 12000, 7500, 18000)
plot(weights, prices)
plot(weights,prices, pch=as.integer(types)) # transforma o vetor com factor em inteiros
# isso vai atribuir simbolos diferentes para cada item
legend("topright", c("gems","gold","silver"), pch=1:3)
# gera legenda automatica, senao cada vez que vc for plotar vai ter de atualizar as legenda
legend("topright", levels(types), pch=1:length(levels(types)))
