-- code contains all the source code files
        -- phase3_main_script.py : Script to perform all the tasks
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
    python phase3_main_script.py -d ../Dataset -t 0

Example query to execute task 1 :-
    python phase3_main_script.py -t 1 -k =30 -l labelled_set1 -u unlabelled_set2

Example query to execute task 2 :-
    python phase3_main_script.py -t 2 -c 5 -l labelled_set2 -u unlabelled_set1

Example query to execute task 3 :-
    python phase3_main_script.py -t 3 -c 5 -m 10 -I "Hand_0008333.jpg Hand_0006183.jpg Hand_0000074" -l labelled_set2

Example query to execute task 4 :-
    python phase3_main_script.py -t 4 -c 5 -T PPR -l labelled_set2 -u unlabelled_set2

Example query to execute task 5 :-
    python phase3_main_script.py -t 5 -i Hand_0000674.jpg -m 20 -L 10 -k 10


Note:- Keep code, <input image directory> and csv at same level
