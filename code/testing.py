from PrincipleComponentAnalysis import PrincipleComponentAnalysis
from ColorMoments import CM
from SIFT import SIFT
from SingularValueDecomposition import SVD
from LatentDirichletAllocation import LDA
from NonNegativeMatrix import NM_F

#md = SIFT("../Hands/Hand_0000002.jpg")

#md.createKMeans("../output/SIFT",30)

pca1 = PrincipleComponentAnalysis()
lda1 = LDA()
svd1 = SVD()
nmf1 = NM_F()

#task1
# pca1.createPCA_KLatentSemantics("LBP", 20)
# pca1.createPCA_KLatentSemantics("LBP", 10)
# lda1.createKLatentSymantics("LBP", 20)
# lda1.createKLatentSymantics("LBP", 10)
# svd1.createKLatentSymantics("LBP", 20)
# svd1.createKLatentSymantics("LBP", 10)
# nmf1.createKLatentSymantics("LBP", 20)
# nmf1.createKLatentSymantics("LBP", 10)

#task2
# pca1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 20, 10)
# pca1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 10, 10)
# lda1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 20, 10)
# lda1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 10, 10)
# svd1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 20, 10)
# svd1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 10, 10)
# nmf1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 20, 10)
# nmf1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 10, 10)

#task3
# pca1.LabelLatentSemantic("dorsal", "HOG", 20)
# pca1.LabelLatentSemantic("left", "CM", 20)
# lda1.LabelLatentSemantic("dorsal", "HOG", 20)
# lda1.LabelLatentSemantic("left", "CM", 20)
# svd1.LabelLatentSemantic("dorsal", "HOG", 20)
# svd1.LabelLatentSemantic("left", "CM", 20)
# nmf1.LabelLatentSemantic("dorsal", "HOG", 20)
# nmf1.LabelLatentSemantic("left", "CM", 20)

#task4
pca1.mSimilarImage_Label("./phase2_Images/Hand_0000200.jpg", "palmar", "LBP", 10, 20)
pca1.mSimilarImage_Label("./phase2_Images/Hand_0011160.jpg", "Access", "SIFT", 10, 10)
lda1.mSimilarImage_Label("./phase2_Images/Hand_0000200.jpg", "palmar", "LBP", 10, 10)
lda1.mSimilarImage_Label("./phase2_Images/Hand_0011160.jpg", "Access", "SIFT", 10, 10)
svd1.mSimilarImage_Label("./phase2_Images/Hand_0000200.jpg", "palmar", "LBP", 10, 10)
svd1.mSimilarImage_Label("./phase2_Images/Hand_0011160.jpg", "Access", "SIFT", 10, 10)
nmf1.mSimilarImage_Label("./phase2_Images/Hand_0000200.jpg", "palmar", "LBP", 10, 10)
nmf1.mSimilarImage_Label("./phase2_Images/Hand_0011160.jpg", "Access", "SIFT", 10, 10)


#task5
# pca1.ImageClassfication("./phase2_Images/Hand_0000928.jpg", "SIFT", 20)




#sd.createKLatentSymantics("LBP",10)


# sd = SVD()
# #sd.createKLatentSymantics("LBP",10)
# sd.mSimilarImage("./phase2_Images/Hand_0000021.jpg", "SIFT", 20, 5)

"""""
md = CM("../Hands/Hand_0000002.jpg")
#md.createKMeans("../output/CM",30)
sd = SVD("../output/CM/Bag_Words.csv")
svd = sd.createKLatentSymantics("../output/CM/Bag_Words.csv", "CM", 10)
sd.mSimilarImage("../phase2", 10, svd,  "../phase2/Hand_0000134.jpg", "CM", 5)
"""""

#lda = LDA()
# lda.createKLatentSymantics("SIFT", 5)
# lda.mSimilarImage("./phase2_Images/Hand_0000021.jpg", "SIFT", 10, 5)

#lda.mSimilarImage_Label("./phase2_Images/Hand_0000209.jpg", "Access", "SIFT", 20, 5)
