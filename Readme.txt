-- code contains all the source code files
        -- phase2_main_script.py : Script to perform all the tasks
        -- ColorMoments.py : Code for CM feature description calculation and matching score
        -- SIFT.py : Code for SIFT feature description calculation and matching score
        -- HOGmain.py: Code for HOG feature description calculation and matching score
        -- LocalBinaryPatterns.py: Code for LBP feature description calculation and matching score
        -- BOW_compute.py: Computes bag of words for run-time image
        -- feature_descriptor.py: Stores feature descriptors and bag of words in DB
        -- LatentDirichletAllocation.py: Handles all the tasks related to LDA
        -- PrincipleComponentAnalysis.py: Handles all the tasks related to PCA
        -- SingularValueDecomposition.py: Handles all the tasks related to SVD
        -- NonNegativeMatrix.py: Handles all the tasks related to NMF
        -- SimilarSubject.py: Handles subject specific tasks
        -- Visualizer.py: Handles the visualizations of the project


-- csv  -- Contains csv files metadata

Initial Set-up:-
    python phase2_main_script.py -d ../Dataset2 -t 0
    mongoimport --port 27018 --db imagedb --type csv --file Desktop/Study/MWDB/ProjectTest/csv/ImageMetadata.csv --headerline
    python phase2_main_script.py -t 9

Example query to execute task 1 :-
    python phase2_main_script.py -M CM -k 20 -T LDA -t 1

Example query to execute task 2 :-
    python phase2_main_script.py -k 10 -m 10 -i Hand_0000111.jpg -T PCA -M HOG -t 2

Example query to execute task 3 :-
    python phase2_main_script.py -k 20 -l left -T LDA -M LBP -t 3

Example query to execute task 4 :-
    python phase2_main_script.py -k 10 -m 10 -i Hand_0000200.jpg -l palmar -T NMF -M LBP -t 4

Example query to execute task 5 :-
    python phase2_main_script.py -k 10 -d ../phase2 -i Hand_0000896.jpg -l right -T SVD -M SIFT -t 5

Example query to execute task 6 :-
    python phase2_main_script.py -s 27 -t 6

Example query to execute task 7 :-
    python phase2_main_script.py -k 10 -t 7

Example query to execute task 8 :-
    python phase2_main_script.py -k 4 -t 8

Note:- Keep code, <input image directory> and csv at same level