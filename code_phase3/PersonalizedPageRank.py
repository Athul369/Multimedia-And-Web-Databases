

def identifyDominantImages(k_sim, k_dom, img_1, img_2, img_3):
    print('Identify dominant images was called.')

    """ Call function to discover the N 'image' elements in the T adjacency matrix """
    t_matrix = createAdjacencyMatrix(k_sim, img_1, img_2, img_3)

    nrml_t_matrix = normalizeMatrix(t_matrix)

    """ We need to select the value of beta (c in research paper). Likely we will want to have a value for beta
        that is slightly above .6 as we should care a little more about topology then the jumps. """

    beta = 0.7

    """ Seed_vector will be a N by 1 matrix. Where all image values are 0 except the 3 query images
        which each will have 1/3 as their value. """
    #seed_vector = 'TBD'

    """ Initialize prev_pi to be same as seed_vector """

    """ Original equation """
    # steady_pi = (1 - beta) * nrml_t_matrix * prev_pi + beta * (seed_vector)

    converged = False

    """ Need to define this while condition in the correct way. """
    while not converged:
        steady_pi = (1 - beta) * nrml_t_matrix * prev_pi + beta * (seed_vector)
        if steady_pi == prev_pi:
            """ If they are the same we have converged """
            converged = True
        else:
            """ If they are different then we need to continue iterating. """
            prev_pi = steady_pi


    """ Once steady_pi has converged we are at our steady state."""
    return getKDominantImages(steady_pi, k_dom)


def createAdjacencyMatrix(k_sim, img_1, img_2, img_3):
    adjMatrix = None
    print('This function will find the k similar images for the 3 query images')
    """ The returned adjacency matrix will hold the N 'image' elements that are within the k similar
        images for at least one of the 3 image queries. Note logic should be added to not add the same
        'image' element twice to the matrix if that 'image' element appears in the k similar images for 
        more than one of our query images. This adjacency matrix also should include the 3 query images. """

    return adjMatrix


def normalizeMatrix(t_matrix):
    normalized = None
    print('This function will Normalize the adjacency matrix by column')
    """ The pairwise relationships between 'image' elements (NODES) will need to be normalized by each
        column in the matrix. In the end the distance values in each column will instead be normalized
        so that the values in each column sum up to 1. """

    return normalized


def getKDominantImages(steady_pi, k_dom):
    k_dom_images = None
    print('This function will return the K most dominant (important) images found from the PPR, RWR algorithm')

    return k_dom_images
