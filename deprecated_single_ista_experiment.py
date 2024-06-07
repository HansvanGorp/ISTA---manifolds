"""
This is deprecated and neeeds to be cleaned up.

It originally came from ista.py as a main function to run the ISTA algorithm.
"""

# %% deprecated functions
def random_analysis_of_ista(ista: ISTA, K: int, nr_points_total: int, nr_points_in_batch: int, max_magnitude: float, A: torch.tensor, save_name: str = "test_figures", save_folder: str = "random_analysis_figures", verbose: bool = False, tqdm_position: int = 0, tqdm_leave: bool = True):
    """
    Creates an analysis of the ISTA module. This is done by looking really into all the axis of y, not just two as specified by the linear projection.
    However, the computational cost of this is high, so we only do this for a subset of the space, namely a hypercube that is centered around the origin.
    Additionally, we do not look into a meshgrid of points, rather we look into a random subset of points.
    """
    # calculate how many batches we need to get the total number of points
    assert nr_points_total % nr_points_in_batch == 0, "nr_points_total should be divisible by nr_points_in_batch"
    nr_batches = nr_points_total // nr_points_in_batch

    # create the save folder if it does not exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    # create the random y data
    M = A.shape[0]
    y = (torch.rand(nr_points_total, M, device="cpu") - 0.5) * 2 * max_magnitude
   
    # run the initials function to get the initial x and jacobian
    x, jacobian = ista.get_initial_x_and_jacobian(nr_points_total, calculate_jacobian = True)

    # put them back on the cpu, because memory is limited
    x = x.cpu()
    jacobian = jacobian.cpu()

    # create an array of nr regions over the iterations
    nr_regions_arrray = torch.zeros(K)

    # loop over the iterations
    for k in tqdm(range(K), position=tqdm_position, leave=tqdm_leave, disable=not verbose, desc="random analysis of ISTA, runnning over folds"):
        unique_entries_current = None
        # loop over the batches
        for i in tqdm(range(nr_batches), position=tqdm_position+1, leave=tqdm_leave, disable= not verbose, desc="random analysis of ISTA, runnning over batches"):
            # extract the y and x data, and jacobian
            y_batch = y[i*nr_points_in_batch:(i+1)*nr_points_in_batch].to(ista.device)
            x_batch = x[i*nr_points_in_batch:(i+1)*nr_points_in_batch].to(ista.device)
            jacobian_batch = jacobian[i*nr_points_in_batch:(i+1)*nr_points_in_batch].to(ista.device)

            # do ista
            with torch.no_grad():
                x_batch, jacobian_batch = ista.forward_at_iteration(x_batch, y_batch, k, jacobian_batch)

            # extract the linear regions from the jacobian
            _, _, unique_entries_new = extract_linear_regions_from_jacobian(jacobian)

            # save the batches back to x and jacobian (on the cpu)
            x[i*nr_points_in_batch:(i+1)*nr_points_in_batch] = x_batch.cpu()
            jacobian[i*nr_points_in_batch:(i+1)*nr_points_in_batch] = jacobian_batch.cpu()

            # push the unqiue entries to the cpu
            unique_entries_new = unique_entries_new.cpu()

            # update the unique entries by comparing it to the current ones, and taking the union of them
            if unique_entries_current is None:
                unique_entries_current = unique_entries_new
            else:
                unique_entries_current = torch.cat((unique_entries_current, unique_entries_new), dim=0)

        # remove duplicates
        unique_entries_current, _ = torch.unique(unique_entries_current, dim=0, return_inverse=True)

        # save the number of regions
        nr_regions_arrray[k] = len(unique_entries_current)

    # plot the number of regions over the iterations
    plt.figure()
    plt.semilogy(nr_regions_arrray,'-', label = "number of linear regions", base = 2)
    plt.semilogy([0,len(nr_regions_arrray)], [nr_points_total,nr_points_total], 'r--', label = "total number of points used", base = 2)
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("number of linear regions")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{save_folder}/{save_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

    return nr_regions_arrray

# %% deprecated main loop
if __name__ == "__main__":
    # set the seed
    torch.manual_seed(0)

    # params
    M = 8
    N = 16
    batch_size = 512
    K_ista =  2**11
    K_lista = 2**10
    mu = 1
    _lambda = 0.1
    device = "cuda:0"

    # grid search for ista?
    perform_grid_search_ista = True
    mus      = np.arange(201)/200 # from 0 to 2, with steps of 0.01
    _lambdas = np.arange(101)/100 # from 0 to 1, with steps of 0.01
    nr_points_for_grid_search = 2**12

    # visual analysis
    nr_points_along_axis = 2**10
    indices_of_projection = (0,1,2)
    margin = 0.5

    # random analysis
    max_magnitude = 2
    nr_points_total    = 2**24
    nr_points_in_batch = 2**21
    
    # knot density analysis
    nr_paths             = 1
    anchor_point_std     = 1
    nr_points_along_path = 2**20  
    path_delta           = 0.001

    # lista data and training
    maximum_sparsity = 4
    x_magnitude = (0, 2)
    nr_iterations_to_train_for = 2048
    end_forgetting_factor = 1
    forgetting_factor = end_forgetting_factor**(1/K_lista)

    # tests
    test_ista_visually      = False
    test_ista_randomly      = False
    test_ista_knot_density  = True

    test_lista_visually     = False
    test_lista_randomly     = False
    test_lista_knot_density = True

    # create a random matrix A
    A = create_random_matrix_with_good_singular_values(M, N)  

    # %% grid search for ISTA
    if perform_grid_search_ista:
        # create a data generator
        data_generator_initialized = lambda: data_generator(A, nr_points_for_grid_search, maximum_sparsity, x_magnitude, N, device)

        # find the best mu and lambda
        mu, _lambda = grid_search_ista(A, data_generator_initialized, mus, _lambdas, K_ista, forgetting_factor = forgetting_factor, device=device)

        # print the best mu and lambda
        print("\ngrid search for ISTA results in:")
        print(f"best mu: {mu}")
        print(f"best lambda: {_lambda}\n")


    # %% test the ISTA module
    if test_ista_visually or test_ista_randomly or test_ista_knot_density:
        # create the ISTA module
        ista = ISTA(A, mu = mu, _lambda = _lambda, K=K_ista, device=device)

        if test_ista_visually:
            # perform the visual analysis on it
            nr_regions_arrray_ista_visual = visual_analysis_of_ista(ista, K_ista, nr_points_along_axis, margin, indices_of_projection, A, save_folder = "ista_figures")
            make_gif_from_figures_in_folder("ista_figures", 5)

        if test_ista_randomly:
            # perform the full analysis on it
            nr_regions_arrray_ista_full = random_analysis_of_ista(ista, K_ista, nr_points_total,  nr_points_in_batch, max_magnitude, A, save_name = "nr_regions_ISTA_random_points")

        if test_ista_knot_density:
            # perform the knot density analysis
            knot_density_ista = knot_density_analysis(ista, K_ista, A, nr_paths = nr_paths,  anchor_point_std = anchor_point_std,
                                                        nr_points_along_path=nr_points_along_path, path_delta=path_delta,
                                                        save_name = "knot_density_ISTA", verbose = True, color = 'tab:blue')

    # %% test the LISTA module
    if test_lista_visually or test_lista_randomly or test_lista_knot_density:
        # create the ISTA module again random initialization
        lista = LISTA(A, mu = mu, _lambda = _lambda, K=K_lista, device=device, initialize_randomly = False)

        # we then train LISTA
        data_generator_initialized = lambda: data_generator(A, batch_size, maximum_sparsity, x_magnitude, N, device)
        lista,_ = train_lista(lista, data_generator_initialized, nr_iterations_to_train_for, forgetting_factor, show_loss_plot = False)

        if test_lista_visually:
            # perform the visual analysis on it
            nr_regions_arrray_lista_learned_visual =  visual_analysis_of_ista(lista, K_lista, nr_points_along_axis, margin, indices_of_projection, A, save_folder = "lista_figures_trained")
            make_gif_from_figures_in_folder("lista_figures_trained", 5)

        if test_lista_randomly:
            # perform the full analysis on it
            nr_regions_arrray_lista_full = random_analysis_of_ista(lista, K_lista, nr_points_total,  nr_points_in_batch, max_magnitude, A, save_name = "nr_regions_LISTA_random_points")

        if test_lista_knot_density:
            # perform the knot density analysis
            knot_density_lista = knot_density_analysis(lista, K_lista, A, nr_paths = nr_paths,  anchor_point_std = anchor_point_std,
                                                         nr_points_along_path=nr_points_along_path, path_delta=path_delta,
                                                         save_name = "knot_density_LISTA", verbose = True, color= 'tab:orange')


    # %% if we plotted both, make a united figure for the number of regions over the iterations
    if test_ista_visually and test_lista_visually:
        plt.figure()
        plt.plot(nr_regions_arrray_ista_visual,'-', label = "ISTA")
        plt.plot(nr_regions_arrray_lista_learned_visual,'-', label = "LISTA after training")
        plt.grid()
        plt.xlabel("iteration")
        plt.ylabel("number of linear regions")
        plt.legend()
        plt.tight_layout()
        plt.savefig("hyperplane_analysis_figures/united_nr_regions_over_iterations.png", dpi=300, bbox_inches='tight')
        plt.savefig("hyperplane_analysis_figures/united_nr_regions_over_iterations.svg", bbox_inches='tight')
        plt.close()

    if test_ista_randomly and test_lista_randomly:
        plt.figure()
        plt.semilogy(nr_regions_arrray_ista_full,'-', label = "ISTA", base = 2)
        plt.semilogy(nr_regions_arrray_lista_full,'-', label = "LISTA after training", base = 2)
        plt.semilogy([0,len(nr_regions_arrray_lista_full)], [nr_points_total,nr_points_total], 'r--', label = "total number of points used", base = 2)
        plt.grid()
        plt.xlabel("iteration")
        plt.ylabel("number of linear regions")
        plt.legend()
        plt.tight_layout()
        plt.savefig("random_analysis_figures/united_nr_regions_over_iterations_random.png", dpi=300, bbox_inches='tight')
        plt.savefig("random_analysis_figures/united_nr_regions_over_iterations_random.svg", bbox_inches='tight')
        plt.close()

    if test_ista_knot_density and test_lista_knot_density:
        K_max = max(K_ista, K_lista)
        folds_ista = np.arange(1,K_ista+1)
        knot_density_ista_mean = knot_density_ista.mean(dim=0)
        folds_lista = np.arange(1,K_lista+1)
        knot_density_lista_mean = knot_density_lista.mean(dim=0)

        plt.figure()
        plt.plot(folds_ista,knot_density_ista_mean,'-', label = "ISTA", c = 'tab:blue')
        plt.plot(folds_lista,knot_density_lista_mean,'-', label = "LISTA", c = 'tab:orange')
        plt.grid()
        plt.xlabel("fold")
        plt.ylabel("knot density")
        plt.legend(loc='best')
        plt.xlim([0,K_max])
        plt.tight_layout()
        plt.savefig("knot_density_figures/united_knot_density_over_iterations.png", dpi=300, bbox_inches='tight')
        plt.savefig("knot_density_figures/united_knot_density_over_iterations.svg", bbox_inches='tight')
        plt.close()
