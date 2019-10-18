from PrincipleComponentAnalysis import PrincipleComponentAnalysis
from ColorMoments import CM
from SIFT import SIFT
from SingularValueDecomposition import SVD
from LatentDirichletAllocation import LDA
from NonNegativeMatrix import NM_F
from SimilarSubject import Subjects

#md = SIFT("../Hands/Hand_0000002.jpg")

#md.createKMeans("../output/SIFT",30)

pca1 = PrincipleComponentAnalysis()
lda1 = LDA()
svd1 = SVD()
nmf1 = NM_F()

#task1
# pca1.createPCA_KLatentSemantics("LBP", 20)
# pca1.createPCA_KLatentSemantics("LBP", 10)
"""
pca1.createPCA_KLatentSemantics("HOG", 20)
pca1.createPCA_KLatentSemantics("HOG", 10)"""
# lda1.createKLatentSymantics("LBP", 20)
# lda1.createKLatentSymantics("LBP", 10)
"""
lda1.createKLatentSymantics("CM", 20)"""
# svd1.createKLatentSymantics("LBP", 20)
# svd1.createKLatentSymantics("LBP", 10)
"""
svd1.createKLatentSymantics("SIFT", 20)
svd1.createKLatentSymantics("SIFT", 10)"""
# nmf1.createKLatentSymantics("LBP", 20)
"""
nmf1.createKLatentSymantics("LBP", 10)"""

#task2
# pca1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 20, 10)
# pca1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 10, 10)
# pca1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "HOG", 10, 10)
pca1.mSimilarImage("./phase2_Images/Hand_0000111.jpg", "HOG", 10, 10)
# lda1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 20, 10)
# lda1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 10, 10)
# svd1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 20, 10)
# svd1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 10, 10)
# nmf1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 20, 10)
nmf1.mSimilarImage("./phase2_Images/Hand_0000200.jpg", "LBP", 10, 10)

#task3
pca1.LabelLatentSemantic("dorsal", "HOG", 20)
# pca1.LabelLatentSemantic("left", "CM", 20)
# lda1.LabelLatentSemantic("dorsal", "HOG", 20)
lda1.LabelLatentSemantic("left", "CM", 20)
# svd1.LabelLatentSemantic("dorsal", "HOG", 20)
# svd1.LabelLatentSemantic("left", "CM", 20)
# nmf1.LabelLatentSemantic("dorsal", "HOG", 20)
# nmf1.LabelLatentSemantic("left", "CM", 20)

#task4
# pca1.mSimilarImage_Label("./phase2_Images/Hand_0000200.jpg", "palmar", "LBP", 10, 20)
# pca1.mSimilarImage_Label("./phase2_Images/Hand_0011160.jpg", "Access", "SIFT", 10, 10)
# lda1.mSimilarImage_Label("./phase2_Images/Hand_0000200.jpg", "palmar", "LBP", 10, 10)
# lda1.mSimilarImage_Label("./phase2_Images/Hand_0011160.jpg", "Access", "SIFT", 10, 10)
# svd1.mSimilarImage_Label("./phase2_Images/Hand_0000200.jpg", "palmar", "LBP", 10, 10)
svd1.mSimilarImage_Label("./phase2_Images/Hand_0011160.jpg", "Access", "SIFT", 10, 10)
nmf1.mSimilarImage_Label("./phase2_Images/Hand_0000200.jpg", "palmar", "LBP", 10, 10)
# nmf1.mSimilarImage_Label("./phase2_Images/Hand_0011160.jpg", "Access", "SIFT", 10, 10)


#task5
# pca1.ImageClassfication("./phase2_Images/Hand_0000111.jpg", "LBP", 20)
pca1.ImageClassfication("./phase2_Images/Hand_0001395.jpg", "SIFT", 20)
lda1.ImageClassfication("./phase2_Images/Hand_0000111.jpg", "LBP", 20)
# lda1.ImageClassfication("./phase2_Images/Hand_0001395.jpg", "SIFT", 20)
# svd1.ImageClassfication("./phase2_Images/Hand_0000111.jpg", "LBP", 20)
svd1.ImageClassfication("./phase2_Images/Hand_0001395.jpg", "SIFT", 10)
nmf1.ImageClassfication("./phase2_Images/Hand_0000111.jpg", "LBP", 10)
# nmf1.ImageClassfication("./phase2_Images/Hand_0001395.jpg", "SIFT", 20)

#task6
#Subjects 27 and 55
s1 = Subjects()
# s1.similar3Subjects('CM', 20, '27')
# s1.similar3Subjects('LBP', 20, '27')
# s1.similar3Subjects('HOG', 20, '27')
# s1.similar3Subjects('SIFT', 20, '27')
# s1.similar3Subjects('CM', 20, '55')
# s1.similar3Subjects('LBP', 20, '55')
# s1.similar3Subjects('HOG', 20, '55')
# s1.similar3Subjects('SIFT', 20, '55')

