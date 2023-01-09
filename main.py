import utils

################ QUESTÃO 2 ################
# ITEM A

lena_gray = utils.Histogram()
lena_gray.read_image('imgs/lena_gray.bmp')

hist_lena_gray = lena_gray.calcHistogram()
lena_gray.show(hist_lena_gray, "Histograma - lena_gray", "Intensidades", "Frequências")

# ITEM B
normal_hist_lena_gray = lena_gray.calcNormalizedHistogram()
lena_gray.show(normal_hist_lena_gray, "Histograma normalizado - lena_gray", "Intensidades", "Probabilidades")

# ITEM C
acum_hist_lena_gray = lena_gray.calcAcumulatedHistogram()
lena_gray.show(acum_hist_lena_gray, "Histograma acumulado - lena_gray", "", "Frequência acumulada")

################ QUESTÃO 3 ################

# ITEM A

image1 = utils.Histogram()

image1.read_image('imgs/image1.png')

hist_image1 = image1.calcHistogram()
image1.show(hist_image1, "Histograma - image1", "Intensidades", "Frequências")

lena_eqHist, lena_eqImg, _ = lena_gray.calcEqualizedHistogram()
img1_eqHist, img1_eqImg, _ = image1.calcEqualizedHistogram()

lena_gray.show(lena_eqHist, "Histograma equalizado - lena_gray", "Intensidades", "Frequências")
image1.show(img1_eqHist, "Histograma equalizado - image1", "Intensidades", "Frequências")

util = utils.Util()

util.save("./results/lena_equalizada.bmp", lena_eqImg)
util.save("./results/image1_equalizada.png", img1_eqImg)

# ITEM B

lena_gray.read_image('results/lena_equalizada.bmp')
image1.read_image('results/image1_equalizada.png')

lena_eqHist, lena_eqImg, _ = lena_gray.calcEqualizedHistogram()
img1_eqHist, img1_eqImg, _ = image1.calcEqualizedHistogram()

lena_gray.show(lena_eqHist, "Histograma equalizado - lena_equalizada", "Intensidades", "Frequências")
image1.show(img1_eqHist, "Histograma equalizado - image1_equalizada", "Intensidades", "Frequências")

util.save("./results/2_lena_equalizada.bmp", lena_eqImg)
util.save("./results/2_image1_equalizada.png", img1_eqImg)

################ QUESTÃO 4 ################

lt = utils.LinearTransformations()

lt.read_image('imgs/lena_gray.bmp')

linear = lt.calcLinear(0.5, 10)
loga = lt.calcLogarithm(16)
exp = lt.calcExponential(1.2)

util.save("./results/4linear.png", linear)
util.save("./results/4logarithm.png", loga)
util.save("./results/4exponential.png", exp)

################ QUESTÃO 5 ################

lena_gray.read_image('imgs/lena_gray.bmp')

lena_hist = lena_gray.calcHistogram()
hist, img = image1.calcSpecifiedHistogram(lena_hist)

image1.show(hist, "Especificação de image1.png com lena_gray", "Intensidades", "Frequências")
util.save("./results/image1_com_lena.png", img)
