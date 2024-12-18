import torch
import numpy as np

def preprocess_data_labels(data, labels):
    # data, labels = batch
    # data = data.to(device)
    # labels = labels.to(device)

    # Sort data samples by labels
    # TODO: Can this be replaced by ConsecutiveLabels ?
    sort = torch.sort(labels)
    # print(sort.indices)
    data = data.squeeze(0)[sort.indices].squeeze(0)
    labels = labels.squeeze(0)[sort.indices].squeeze(0)
    return data, labels

def get_support_noise_query_indices(ways, shot, query_num):
    # Init support and noise arrays
    support_indices = np.zeros(ways * (2 * shot + query_num), dtype=bool)
    noise_indices = support_indices.copy()

    # Marker for beginning of each of the ways
    selection = np.arange(ways) * (2 * shot + query_num)

    # Mark support indices, starting from the beginning of each class
    for offset in range(shot):
        support_indices[selection + offset] = True
        noise_indices[selection + shot + offset] = True
    # Query indices are those that aren't support or noise
    query_indices = ~(support_indices | noise_indices)

    # Convert to torch
    return {
        "support": torch.from_numpy(support_indices),
        "noise": torch.from_numpy(noise_indices),
        "query": torch.from_numpy(query_indices),
    }

def gen_swap_indices(ways, num_noise_samples, num_clean_shots, indices_to_change):
    shot = num_noise_samples // ways

    # Track unpicked options
    available = np.arange(num_noise_samples)
    available_KN = available.reshape(ways, shot)

    swap_indices = {}

    # permute for loop to avoid bias of class 0 picking first
    class_order = np.arange(ways)
    np.random.shuffle(class_order)
    for c in class_order:
        # Remove samples from class c from options
        available_c = [i for i in available if i not in available_KN[c]]

        # Track number of noise samples from each class
        noise_class_count = np.zeros(ways)

        swap_indices_c = []

        for i in range(len(indices_to_change[c])):
            # Random pick, if there are enough choices
            if len(available_c) == 0:
                return None

            # Pick random sample idx from available choices; remove from availabe options
            noise_choice_i = np.random.choice(available_c)
            swap_indices_c.append(noise_choice_i)
            available_c.remove(noise_choice_i)

            # Find class of picked sample and increment class counter
            noise_choice_class_i = noise_choice_i // shot
            noise_class_count[noise_choice_class_i] += 1

            # Limit noise samples per class to ensure clean class has plurality
            # If class has 1 less than the number of clean class, remove other samples from choices
            if (noise_class_count[noise_choice_class_i] + 1) >= num_clean_shots:
                available_c = [
                    i
                    for i in available_c
                    if i not in available_KN[noise_choice_class_i]
                ]

        swap_indices[c] = swap_indices_c

        # Update unpicked options
        available = np.setdiff1d(available, swap_indices[c])

    # Put swap indices back in order
    swap_indices = np.vstack([swap_indices[c] for c in range(ways)])

    return swap_indices

def gen_valid_swap_indices(ways, shot, indices_to_change):
    num_noise_samples = ways * shot
    num_clean_shots = shot - len(indices_to_change[0])

    while True:
        swap_indices = gen_swap_indices(
            ways, num_noise_samples, num_clean_shots, indices_to_change
        )

        if swap_indices is not None:
            break

    return swap_indices

def gen_derangement(n):
    """
    Generates a derangement (random permutation without any fixed points)
    """
    in_order = np.arange(n, dtype=np.int16)
    while True:
        derangement_candidate = np.random.permutation(n)
        if 0 not in derangement_candidate - in_order:
            break
    return derangement_candidate

def add_noise(
    data,
    labels,
    mask_indices,
    ways,
    noise_fraction,
    noise_type="sym_swap",
    outlier_data=None,
):
    """
    Adds either label swap noise (symmetric or paired) or outlier noise
    """
    # Calculate number of noisy samples
    num_idx = np.arange(data.shape[0], dtype=np.int16)[mask_indices["support"]]
    shot = int(len(num_idx) / ways)
    noise_num = int(round(noise_fraction * shot))

    # Select the indices of samples to noise
    indices_to_change = np.empty((0, noise_num), dtype=np.int16)
    for i in range(ways):
        class_i_idx = num_idx[i * shot : (i + 1) * shot]
        indices_to_change_i = np.random.choice(class_i_idx, noise_num, replace=False)
        indices_to_change = np.vstack((indices_to_change, indices_to_change_i))
    indices_to_change_flat = indices_to_change.flatten()

    # Make copy of data and labels
    noised_data = torch.clone(data)
    noised_labels = torch.clone(labels)

    # Replace clean data with noisy data at selected positions

    # print(noise_type)

    if noise_type == "sym_swap":
        # Swap data positions of selected samples, ensuring plurality of clean class
        swap_indices = gen_valid_swap_indices(ways, shot, indices_to_change)
        # print(mask_indices, swap_indices)
        noised_data[indices_to_change_flat] = data[mask_indices["noise"]][
            swap_indices.flatten()
        ]
        # print('indices_to_change_flat:', indices_to_change_flat)
        # print('mask_indices["noise"]:', mask_indices["noise"])
        # print('noised_labels[indices_to_change_flat]:', noised_labels[indices_to_change_flat])
        # print('labels[mask_indices["noise"]][swap_indices.flatten()]:', labels[mask_indices["noise"]][
        #     swap_indices.flatten()
        # ])
        noised_labels[indices_to_change_flat] = labels[mask_indices["noise"]][
            swap_indices.flatten()
        ]

    elif noise_type == "pair_swap":
        # Randomly select class pairs for noise
        drng = gen_derangement(ways)

        # Reorganize data and labels by class
        noisy_data = data[mask_indices["noise"]].reshape(
            ways, shot, data.shape[1], data.shape[2], data.shape[3]
        )
        noisy_labels = labels[mask_indices["noise"]].reshape(ways, shot)

        # Swap in noisy data from noisy samples
        for c in range(ways):
            noised_data[indices_to_change[c]] = noisy_data[drng[c]][:noise_num]
            noised_labels[indices_to_change[c]] = noisy_labels[drng[c]][:noise_num]

    # elif noise_type == "outlier":
    #     # Randomly select outlier indices
    #     # Draw new indices rather than reuse indices_to_change so outlier classes don't line up
    #     outlier_select_indices = np.random.choice(
    #         len(outlier_data), len(indices_to_change_flat), replace=False
    #     )
    #     noised_data[indices_to_change_flat] = outlier_data[outlier_select_indices]
    #     noised_labels[indices_to_change_flat] = OUTLIER

    else:
        raise NotImplementedError

    # Location of noised data
    noise_positions = np.zeros(data.shape[0])
    noise_positions[indices_to_change_flat] = 1

    return noised_data, noised_labels, noise_positions

def prepare_data(args, data, noisy_type="pair_swap", percent=0.0):
    labels = torch.arange(args.way, dtype=torch.int16).repeat(args.query+args.shot*2)
    labels = labels.type(torch.LongTensor).cuda()

    data, labels = preprocess_data_labels(data, labels)
    # Separate query and support
    mask_indices = get_support_noise_query_indices(args.way, args.shot, args.query)

    # Add noise
    noise_positions = np.zeros(data.shape[0])
    # print(len(noise_positions))
    if percent > 0:
        data, labels, noise_positions = add_noise(
            data,
            labels,
            mask_indices,
            5,
            percent,
            noisy_type,
            None,
        )
    
    data = data[~mask_indices["noise"]]

    indices_final = []
    for i in range(20):
        for j in range(5):
            indices_final.append(j*20 + i)
    indices_final = torch.tensor(indices_final).cuda()

    data = data.squeeze(0)[indices_final].squeeze(0)

    return data

