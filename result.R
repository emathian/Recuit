# coinditions initiales aléatoire
# k and f 40 simulation
c1 = c( 0. ,26. ,26. ,34. ,33. ,32. ,31. ,27. ,31. ,30. ,25. ,30. ,28. ,31. ,28. ,32. ,28. ,32.,
        32. ,28. ,30. ,30. ,32. ,34. ,29. ,32. ,27. ,28. ,30. ,29.)
c2 = c(32. ,29. ,29. ,29. ,35. ,35. ,29. ,29. ,27. ,27. ,38. ,31. ,32. ,25. ,31. ,28. ,33. ,31.,
       35. ,29. ,32. ,27. ,30. ,32. ,33. ,34. ,29. ,30. ,31. ,31.)

c1 = c1[2:30 ]
shapiro.test(c1)
shapiro.test(c2)
boxplot(c1,c2)
var.test(c1,c2)
t.test(c1, c2)
t.test(c1, c2, 'less')
vk = seq(0.5, 29.5 , 0.5)
T1  = data.frame(val = c(c1,c2),
                 k = vk,
                 classe = c(rep('a[0,7.5]', 14),
                            rep('b[8,15]', 15),
                            rep('c[15.5,23]', 15),
                            rep('d[23,30]', 15))
)
boxplot(T1$val~T1$classe, col=c('darkblue', 'darkred', 'gold', 'darkgreen'))
bartlett.test(T2$val~T2$classe)
kruskal.test(T1$val~T1$classe)


# On ne peut pas rejeter h0 :(

#kp and f 40 simulation
c3 = c(40. ,38. ,37. ,34. ,38. ,32. ,32. ,28. ,28. ,32. ,30. ,35. ,30. ,32. ,31. ,29. ,24. ,37.,
       28. ,29. ,34. ,32. ,29. ,28. ,31)
c4 = c(32. ,27. ,31. ,31. ,29. ,31. ,31. ,31. ,27. ,34. ,34. ,33. ,34. ,26. ,29. ,33. ,30. ,28.,
        29. ,30. ,33. ,30. ,29. ,28. ,29.)

shapiro.test(c3)
shapiro.test(c4)
var.test(c3,c4)
# L'homoscédasticité n'est pas respectée
boxplot(c3,c4)
plot(c3)
plot(c4)


vkp = seq(0, 0.98 , 0.02)
T2  = data.frame(val = c(c3,c4),
                 k = vkp,
                 classe = c(rep('a[0,22]', 12),
                            rep('b (0.22,0.48]', 13),
                            rep('c(0.48,0.72]', 12),
                            rep('d(0.72,1)', 13))
)
bartlett.test(T2$val~T2$classe)
boxplot(T2$val~T2$classe, col=c('darkblue', 'darkred', 'gold', 'darkgreen'))
kruskal.test(T2$val~T2$classe)
wilcox.test(T2$val[T2$classe=='a[0,22]'],T2$val[T2$classe=='b (0.22,0.48]'])
# La classe 1 a de meilleur resultats

# k and g 40 simulation
c5 = c(0. ,26. ,24. ,21. ,20. ,17. ,20. ,26. ,19. ,24. ,26. ,18. ,23. ,22. ,18. ,23. ,18. ,28.,
       24. ,25. ,23. ,26. ,27. ,21. ,18. ,27. ,23. ,20. ,23. ,23.)
c6 = c(19. ,24. ,24. ,23. ,29. ,23. ,22. ,21. ,24. ,25. ,25. ,20. ,23. ,31. ,22. ,25. ,28. ,26.,
       19. ,23. ,24. ,22. ,20. ,27. ,24. ,24. ,23. ,17. ,23. ,19.)


c5 = c5[2:30 ]
shapiro.test(c5)
shapiro.test(c6)
boxplot(c5,c6)
var.test(c5,c6)
t.test(c5, c6)

vk = seq(0.5, 29.5 , 0.5)
T1g  = data.frame(val = c(c5,c6),
                 k = vk,
                 classe = c(rep('a[0,7.5]', 14),
                            rep('b[8,15]', 15),
                            rep('c[15.5,23]', 15),
                            rep('d[23,30]', 15))
)
boxplot(T1g$val~T1g$classe, col=c('darkblue', 'darkred', 'gold', 'darkgreen'))
bartlett.test(T2$val~T2$classe)
kruskal.test(T1$val~T1$classe)

#kp and g 40 simulation
c7 = c(27. ,22. ,28. ,25. ,29. ,17. ,19. ,27. ,26. ,22. ,24. ,19. ,17. ,26. ,24. ,26. ,25.,
       31. ,20. ,20. ,18. ,21. ,23. ,18.)
c8 = c(20. ,22. ,19. ,22. ,22. ,25. ,26. ,20. ,19. ,23. ,21. ,17. ,17. ,21. ,22. ,22. ,15. ,25.,
       22. ,21. ,23. ,24. ,19. ,24. ,22.)

shapiro.test(c7)
shapiro.test(c8)
var.test(c7,c8)
# L'homoscédasticité n'est pas respectée ni la normalité
boxplot(c7,c8)
plot(c7)
plot(c8)


vkp = seq(0.02, 0.98 , 0.02)
T2g  = data.frame(val = c(c7,c8),
                 k = vkp,
                 classe = c(rep('a[0,22]', 11),
                            rep('b (0.22,0.48]', 13),
                            rep('c(0.48,0.72]', 12),
                            rep('d(0.72,1)', 13))
)
bartlett.test(T2g$val~T2g$classe) # l'homogénité des variances n'est pas réespectée
kruskal.test(T2g$val~T2g$classe)
boxplot(T2g$val~T2g$classe, col=c('darkblue', 'darkred', 'gold', 'darkgreen'))
wilcox.test(T2g$val[T2g$classe=='a[0,22]'],T2g$val[T2g$classe=='b (0.22,0.48]'])
# La classe 1 a de meilleur resultats




# Condition initiales fixées à 0
# k and f 40 simulation
c1f  = c( 0. ,27. ,24. ,26. ,24. ,32. ,32. ,34. ,28. ,31. ,32. ,29. ,23. ,28. ,32. ,34. ,30. ,30.,
          34. ,33. ,33. ,30. ,29. ,28. ,27. ,29. ,32. ,32. ,30. ,30.)
c2f = c(28. ,26. ,25. ,29. ,33. ,33. ,26. ,31. ,33. ,33. ,28. ,29. ,27. ,31. ,28. ,35. ,32. ,33.,
        31. ,29. ,31. ,29. ,31. ,30. ,33. ,33. ,32. ,29. ,30. ,30.)


c1f = c1f[2:30 ]
shapiro.test(c1f)
shapiro.test(c2f)
boxplot(c1f,c2f)
var.test(c1f,c2f)
t.test(c1f, c2f)
t.test(c1f, c2f, 'less')
vk = seq(0.5, 29.5 , 0.5)
T1F  = data.frame(val = c(c1f,c2f),
                 k = vk,
                 classe = c(rep('a[0,7.5]', 14),
                            rep('b[8,15]', 15),
                            rep('c[15.5,23]', 15),
                            rep('d[23,30]', 15))
)
boxplot(T1F$val~T1F$classe, col=c('darkblue', 'darkred', 'gold', 'darkgreen'))
bartlett.test(T1F$val~T1F$classe)  # Absence d'homoscédasticité
kruskal.test(T1F$val~T1F$classe)



#kp and f 40 simulation
c3f= c(40. ,38. ,32. ,31. ,33. ,30. ,30. ,29. ,30. ,34. ,28. ,32. ,32. ,31. ,32. ,32. ,30. ,35.,
       29. ,32. ,28. ,32. ,28. ,31. ,31.)
c4f = c(33. ,28. ,28. ,32. ,31. ,35. ,30. ,32. ,28. ,32. ,32. ,31. ,30. ,26. ,32. ,24. ,31. ,33.,
        29. ,31. ,30. ,32. ,32. ,29. ,30.)

shapiro.test(c3f)
shapiro.test(c4f)
var.test(c3,c4f)
# L'homoscédasticité n'est pas respectée
boxplot(c3f,c4f)
plot(c3f)
plot(c4f)


vkp = seq(0, 0.98 , 0.02)
T2F  = data.frame(val = c(c3f,c4f),
                 k = vkp,
                 classe = c(rep('a[0,22]', 12),
                            rep('b (0.22,0.48]', 13),
                            rep('c(0.48,0.72]', 12),
                            rep('d(0.72,1)', 13))
)
bartlett.test(T2F$val~T2F$classe)
boxplot(T2F$val~T2F$classe, col=c('darkblue', 'darkred', 'gold', 'darkgreen'))
kruskal.test(T2F$val~T2F$classe)


#k and g 40 simulation
c5f= c( 0. ,20. ,19. ,20. ,22. ,27. ,25. ,20. ,23. ,18. ,19. ,23. ,21. ,23. ,26. ,27. ,26. ,18.,
        23. ,25. ,26. ,24. ,23. ,24. ,23. ,24. ,23. ,27. ,30. ,20)
c6f = c(24. ,24. ,25. ,23. ,22. ,26. ,22. ,26. ,21. ,30. ,26. ,21. ,23. ,18. ,24. ,17. ,29. ,22.,
        22. ,25. ,22. ,23. ,29. ,22. ,23. ,27. ,24. ,23. ,25. ,22)

c5f = c5f[2:30 ]
shapiro.test(c5f)
shapiro.test(c6f)
boxplot(c5f,c6f)
var.test(c5f,c6f)
t.test(c5f, c6f)

vk = seq(0.5, 29.5 , 0.5)
T1gF  = data.frame(val = c(c5f,c6f),
                  k = vk,
                  classe = c(rep('a[0,7.5]', 14),
                             rep('b[8,15]', 15),
                             rep('c[15.5,23]', 15),
                             rep('d[23,30]', 15))
)
boxplot(T1gF$val~T1gF$classe, col=c('darkblue', 'darkred', 'gold', 'darkgreen'))
bartlett.test(T1gF$val~T1gF$classe)
kruskal.test(T1gF$val~T1gF$classe)
wilcox.test(T1gF$val[T1gF$classe=='a[0,7.5]'],T1gF$val[T1gF$classe=='b[8,15]'])
#kp and g 40 simulation
c7f= c(39. ,28. ,25. ,26. ,17. ,27. ,30. ,25. ,19. ,23. ,23. ,30. ,23. ,24. ,23. ,21. ,23. ,26.,
       20. ,24. ,24. ,24. ,23. ,28. ,14)
c8f = c(15. ,23. ,26. ,18. ,25. ,24. ,21. ,27. ,24. ,20. ,25. ,19. ,22. ,24. ,22. ,21. ,20. ,27.,
        22. ,24. ,24. ,19. ,20. ,19. ,22.)
c7f = c7f[2:25]
shapiro.test(c7f)
shapiro.test(c8f)
var.test(c7f,c8f)
# L'homoscédasticité n'est pas respectée ni la normalité
boxplot(c7f,c8f)
plot(c7f)
plot(c8f)


vkp = seq(0.02, 0.98 , 0.02)
T2gF  = data.frame(val = c(c7f,c8f),
                  k = vkp,
                  classe = c(rep('a[0,22]', 11),
                             rep('b (0.22,0.48]', 13),
                             rep('c(0.48,0.72]', 12),
                             rep('d(0.72,1)', 13))
)
bartlett.test(T2gF$val~T2gF$classe) # l'homogénité des variances n'est pas réespectée
boxplot(T2gF$val~T2gF$classe, col=c('darkblue', 'darkred', 'gold', 'darkgreen'))
kruskal.test(T2gF$val~T2gF$classe)

