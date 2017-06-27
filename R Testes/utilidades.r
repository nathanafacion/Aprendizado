pt_data = read.csv("C:\\Users\\natha\\Desktop\\Unicamp Primeiro Semestre\\Aprendizado\\example.csv", stringsAsFactors = FALSE)

str(pt_data)

#Analise estatistica
summary(pt_data$X3)

#Minimo e maximo
range(pt_data$X3)

#Diferenca entre minimo e maximo
diff(range(pt_data$X3))

#diferenca entre quadrante q25 e q75
IQR(range(pt_data$X3))

#quartis
quantile(pt_data$X3)

#boxplots
boxplot(pt_data$X3, main=" Testando boxplot X3", ylab="X3")

#histograma
hist(pt_data$X3, main=" Testando boxplot X3", xlab="X3")

#desvio padrao

sd(pt_data$X3)

# Verificar se duas variaveis sao correlacionadas

plot(x=pt_data$X3,y=pt_data$X35, main="Testando variaveis", xlab="X3", ylab="X35")

install.packages("gmodels")