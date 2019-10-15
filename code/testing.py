from NonNegativeMatrix import NMF
from SingularValueDecomposition import SVD
from ColorMoments import CM
from SIFT import SIFT

#md = SIFT("../Hands/Hand_0000002.jpg")

#md.createKMeans("../output/SIFT",30)

sd = SVD()
#sd.createKLatentSymantics("LBP",10)
#sd.mSimilarImage("../phase2/Hand_0001804.jpg", "LBP", 10, 5)
nf=NMF()
nf.createKLatentSymantics("LBP",20)

"""""

md = CM("../Hands/Hand_0000002.jpg")

#md.createKMeans("../output/CM",30)

sd = SVD("../output/CM/Bag_Words.csv")
svd = sd.createKLatentSymantics("../output/CM/Bag_Words.csv", "CM", 10)
sd.mSimilarImage("../phase2", 10, svd,  "../phase2/Hand_0000134.jpg", "CM", 5)
"""""